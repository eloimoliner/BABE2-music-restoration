import pickle
import torch
import torchaudio
import numpy as np
import utils.dnnlib as dnnlib
import os
import utils.logging as utils_logging
import copy
import utils.blind_bwe_utils as blind_bwe_utils
import wandb
import soundfile as sf

class Evaluator():
    def __init__(
        self, args=None, network=None, diff_params=None, test_set=None, device=None, it=None
    ):
        self.args=args
        self.network=torch.compile(network)

        self.diff_params=copy.copy(diff_params)
        self.device=device
        #choose gpu as the device if possible
        if self.device is None:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network=network

        torch.backends.cudnn.benchmark = True

        if it is None:
            self.it=0

        self.setup_sampler()

        self.use_wandb=False #hardcoded for now

    def setup_sampler(self):
        self.sampler=dnnlib.call_func_by_name(func_name=self.args.tester.sampler_callable, model=self.network,  diff_params=self.diff_params, args=self.args) #rid is used to log some extra information

    def load_checkpoint(self, path):

        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict['ema'])
        self.it=state_dict['it']
        self.LTAS_ref=state_dict['LTAS'].to(self.device)

               
    def apply_LTAS_init(self, recording):

            nfft=self.args.tester.blind_bwe.LTAS_fft
            freqs = np.fft.fftfreq(nfft, 1/self.args.exp.sample_rate)[0:nfft//2+1]
            freqs[-1]=self.args.exp.sample_rate/2
            freqs=torch.tensor(freqs, dtype=torch.float32).to(recording.device)

            LTAS_smooth=blind_bwe_utils.smooth_LTAS(self.LTAS_ref,freqs,Noct=3)

            max=LTAS_smooth.max()
            correction=1/max #that correction may not be necessary, but I will keep it for now
            LTAS_smooth=LTAS_smooth*correction
            LTAS_smooth=LTAS_smooth.to(recording.device)

            LTAS_y, std=blind_bwe_utils.compute_LTAS(x=recording.clone(), sample_rate=self.args.exp.sample_rate, normalize=1 ,nfft=nfft,win_length=nfft//2, hop_length=nfft//4)

            LTAS_y_smooth=blind_bwe_utils.smooth_LTAS(LTAS_y, freqs, Noct=3)*correction.to(LTAS_y.device)

            diff=torch.clamp(10*torch.log10(LTAS_y_smooth) -10*torch.log10(LTAS_smooth),min=-20)

            #sorry for the hardcoding
            win_length = nfft//2
            hop_length = nfft//4

            X=torch.stft(recording, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length), return_complex=True)
            diff_lin=torch.pow(10, diff/20)
            X/=diff_lin.unsqueeze(-1)

            x_init=torch.istft(X, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length), length=recording.shape[0])

            x_init=std*x_init/x_init.std()

            return x_init

    def segment_signal(self,recording, seg_len, overlap):
                    L=recording.shape[-1]
                    ix_start=[]
                    ix_end=[]
                    segs=[]
                    ix=0
                    while ix+seg_len<L:
                        segs.append(torch.Tensor(recording[...,ix:ix+seg_len]))
                        ix_start.append(ix)
                        ix_end.append(ix+seg_len)
                        ix+=seg_len-overlap
                    
                    #add the last segment with zeros if necessary
                    segs.append(torch.cat((torch.Tensor(recording[...,ix::]).to(self.device), torch.zeros((seg_len-(L-ix),), device=self.device)), -1))
                    ix_start.append(ix)
                    ix_end.append(ix+seg_len)
                    expanded_size=recording.shape[-1]+seg_len-(L-ix)
                    return segs, ix_start, ix_end, expanded_size

    def evaluate_single_recording(self):

        recording_path=self.args.tester.evaluation.single_recording

        try:
            recording,fs=sf.read(recording_path)
        except:
            #maybe it is a relative path
            #find my path
            path=os.path.dirname(os.path.realpath(__file__))
            #remove last folder from the path
            path=os.path.split(path)[0]
            print("path",path)
            recording_path=os.path.join(path,recording_path)
            recording,fs=sf.read(recording_path)


        print("recording",recording.shape)
        #stereo to mono
        if len(recording.shape)>1:
            recording=recording.mean(axis=1)

        if fs!=self.args.exp.sample_rate: 
            print("sample rate of the recording does not match the sample rate of the model")
            #resample
            recording=torchaudio.functional.resample(torch.Tensor(recording), fs, self.args.exp.sample_rate)
        else:
            recording=torch.Tensor(recording)

        print("filename",recording_path)
        #segment the recording
        seg_mode=self.args.tester.evaluation.segment_placement
        seg_len=self.args.tester.evaluation.segment_length
        if seg_len != self.args.exp.audio_len:
            #necessary because of the CQT
            self.sampler.model.setup_seg_len(seg_len)

        num_segments_batch=self.args.tester.evaluation.num_segments_batch

        if self.args.tester.evaluation.LTAS_init:
            x_init=self.apply_LTAS_init(recording)

            path, basename=os.path.split(recording_path)
            sf.write(os.path.join(path, basename+"LTAS.wav"), x_init, self.args.exp.sample_rate)

            if self.args.tester.evaluation.LTAS_as_y:
                recording=x_init

        overlap=int(self.args.tester.evaluation.overlap*seg_len)


        #divide the recording in segments of length seg_len (with overlap)
        segs, ix_start, ix_end,  expanded_size=self.segment_signal(recording, seg_len, overlap)
        result=torch.zeros((expanded_size,), device="cpu")
        result_mask=torch.zeros((expanded_size,), device="cpu")
        #calculate std of the recording, ignoring the silent passages
        std_s=[segs[i].std() for i in range(0, len(segs))]
        #take the median std
        std_orig=torch.median(torch.Tensor(std_s))

        segs=[self.args.tester.blind_bwe.sigma_norm*segs[i]/std_orig for i in range(0, len(segs))]

        processed_segs=[0 for i in range(0, len(segs))]

                
        y=torch.zeros((1, seg_len))
                    
        y[0]=segs[0].to(torch.float32)

        if self.args.tester.evaluation.process_complete_mode=="Time-Invariant":
                if seg_mode=="middle":
                    #select n segments from the middle of the recording. if there is more than one segment, then divide the recording in n parts of equal length, and select the middle segment of each part
                    y=torch.zeros((num_segments_batch, seg_len))
                    
                    ix_start_blind=[]
                    ix_end_blind=[]
                    print(num_segments_batch)
                    for i in range(0, num_segments_batch):
                        ix=int(len(segs)/num_segments_batch*i+len(segs)/num_segments_batch/2)
                        y[i,...]=segs[ix].to(torch.float32)
                        processed_segs[ix]=1
                        ix_start_blind.append(ix_start[ix])
                        ix_end_blind.append(ix_end[ix])
                else:
                    raise NotImplementedError

        if self.args.tester.evaluation.LTAS_init:

            assert x_init.shape==recording.shape, "x_init and recording have different shapes: "+str(x_init.shape)+" and "+str(recording.shape)

            segs_init, ix_start, ix_end,  expanded_size=self.segment_signal(x_init, seg_len, overlap)

            #take the median std
            std_orig=torch.median(torch.Tensor(std_s))
    
            segs_init=[self.args.tester.blind_bwe.sigma_norm*segs_init[i]/std_orig for i in range(0, len(segs))]
    

            #raise ValueError("blef")

            if self.args.tester.evaluation.process_complete_mode=="Block-Autoregressive":
                x_seg_init=torch.zeros((1, seg_len))
                        
                x_seg_init[0]=segs_init[0].to(torch.float32)

            if self.args.tester.evaluation.process_complete_mode=="Time-Invariant":
            
                x_seg_init=torch.zeros((num_segments_batch, seg_len))
                for i in range(0, num_segments_batch):
                        ix=int(len(segs)/num_segments_batch*i+len(segs)/num_segments_batch/2)
                        x_seg_init[i]=segs_init[ix].to(torch.float32)


                    
        path, basename=os.path.split(recording_path)
                
        y=y.to(self.device)

        if self.args.tester.evaluation.LTAS_init:
            x_init=x_seg_init.to(self.device)
        else:
            x_init=None
        outputs=self.sampler.predict_blind_bwe(y, x_init=x_init)
        
        pred, estimated_filter =outputs



        if self.args.tester.evaluation.process_complete_mode=="Block-Autoregressive":
                #add estimated_filter into a list
                estimated_filters=[estimated_filter]

                for i in range(0, pred.shape[0]):
                    result[ix_start[0]:ix_end[0]]=pred[i].cpu()
                    result_mask[ix_start[0]:ix_end[0]]=1

                #save result    
                sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), std_orig*result.cpu().numpy()/self.args.tester.blind_bwe.sigma_norm, self.args.exp.sample_rate)

                for i in range(1, len(segs)):
                    #process segs one by one
                    if processed_segs[i]==0:
                        y=segs[i].to(self.device)
                        if self.args.tester.evaluation.LTAS_init:
                            x_seg_init=segs_init[i].to(self.device)

                        y_masked=result[ix_start[i]:ix_end[i]].clone()
                        y_masked=y_masked.to(self.device)
                        mask=result_mask[ix_start[i]:ix_end[i]].clone()
                        mask=mask.to(self.device)
                        rid=False
                        out=self.sampler.predict_blind_bwe_AR(y, y_masked, mask=mask, x_init=x_seg_init if self.args.tester.evaluation.LTAS_init else None)
                        pred, estimated_filter=out
                        estimated_filters.append(estimated_filter)
                        
                        print("masks",mask[...,0], mask[...,-1])
                        if mask[...,0]==1:
                            print("hann in left")
                            #apply a hann window on the overlapping region
                            #find first occurrence of 0 in the mask
                            first_0=torch.where(mask==0)[0][0]
                            hann_window=torch.hann_window(first_0*2, device=self.device)
                            hann_left=hann_window[0:first_0]
                            hann_right=hann_window[first_0::]
                            pred[...,0:first_0]=pred[...,0:first_0]*hann_left+y_masked[0:first_0]*hann_right

                        if mask[...,-1]==1:
                            print("hann in right")
                            #find last occurrence of 0 in the mask
                            last_0=torch.where(mask==0)[0][-1]
                            size=seg_len-last_0
                            hann_window=torch.hann_window((size)*2, device=self.device)
                            hann_left=hann_window[0:size]
                            hann_right=hann_window[size::]
                            pred[...,seg_len-size::]=pred[...,seg_len-size::]*hann_right+y_masked[...,seg_len-size::]*hann_left
                            

                        result_mask[ix_start[i]:ix_end[i]]=1
                        result[ix_start[i]:ix_end[i]]=pred[0].cpu()
                        #save result
                        sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), std_orig*result.cpu().numpy()/self.args.tester.blind_bwe.sigma_norm, self.args.exp.sample_rate)

                result=result[0:recording.shape[-1]]
                sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), std_orig*result.cpu().numpy()/self.args.tester.blind_bwe.sigma_norm, self.args.exp.sample_rate)

                #save list of estimated_filters as a pkl file, as this will be used for the whole recording
                path, basename=os.path.split(recording_path)
                with open(os.path.join(path, basename+".filter_"+str(self.args.id)+".pkl"), "wb") as f:
                    pickle.dump(estimated_filters, f)

        elif self.args.tester.evaluation.process_complete_mode=="Time-Invariant":

                #save estimated_filter as a pkl file, as this will be used for the whole recording
                path, basename=os.path.split(recording_path)
                with open(os.path.join(path, basename+".filter_"+str(self.args.id)+".pkl"), "wb") as f:
                    pickle.dump(estimated_filter, f)

                for i in range(0, pred.shape[0]):
                    result[ix_start_blind[i]:ix_end_blind[i]]=pred[i].cpu()
                    result_mask[ix_start_blind[i]:ix_end_blind[i]]=1
                #save result    
                sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), self.args.dset.sigma_data*result.cpu().numpy(), self.args.exp.sample_rate)

                for i in range(0, len(segs)):
                    #process segs one by one
                    if processed_segs[i]==0:
                        y=segs[i].to(self.device)
                        y_masked=result[ix_start[i]:ix_end[i]].clone()
                        y_masked=y_masked.to(self.device)
                        mask=result_mask[ix_start[i]:ix_end[i]].clone()
                        mask=mask.to(self.device)
                        rid=False
                        out=self.sampler.predict_bwe_AR(y, y_masked, estimated_filter, "BABE2",rid=rid,  test_filter_fit=False, compute_sweep=False, mask=mask)
                        pred=out
                        
                        print("masks",mask[...,0], mask[...,-1])
                        if mask[...,0]==1:
                            print("hann in left")
                            #apply a hann window on the overlapping region
                            #find first occurrence of 0 in the mask
                            first_0=torch.where(mask==0)[0][0]
                            hann_window=torch.hann_window(first_0*2, device=self.device)
                            hann_left=hann_window[0:first_0]
                            hann_right=hann_window[first_0::]
                            pred[...,0:first_0]=pred[...,0:first_0]*hann_left+y_masked[0:first_0]*hann_right

                        if mask[...,-1]==1:
                            print("hann in right")
                            #find last occurrence of 0 in the mask
                            last_0=torch.where(mask==0)[0][-1]
                            size=seg_len-last_0
                            hann_window=torch.hann_window((size)*2, device=self.device)
                            hann_left=hann_window[0:size]
                            hann_right=hann_window[size::]
                            pred[...,seg_len-size::]=pred[...,seg_len-size::]*hann_right+y_masked[...,seg_len-size::]*hann_left
                            

                        result_mask[ix_start[i]:ix_end[i]]=1
                        result[ix_start[i]:ix_end[i]]=pred[0].cpu()
                        #save result
                        sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), std_orig*result.cpu().numpy()/self.args.tester.blind_bwe.sigma_norm, self.args.exp.sample_rate)

                result=result[0:recording.shape[-1]]
                sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), std_orig*result.cpu().numpy()/self.args.tester.blind_bwe.sigma_norm, self.args.exp.sample_rate)

        else:
                raise NotImplementedError

        return

    def sample_unconditional(self):
        shape=[self.args.tester.unconditional.num_samples, self.args.tester.unconditional.audio_len]
        #TODO assert that the audio_len is consistent with the model
        rid=False
        outputs=self.sampler.predict_unconditional(shape, self.device, rid=rid)
        if rid:
            preds, data_denoised, t=outputs
            fig_animation_sig=utils_logging.diffusion_spec_animation(self.paths["blind_bwe"],data_denoised, t[:-1], self.args.logging.stft,name="unconditional_signal_generation")
        else:
            preds=outputs

        self.log_audio(preds*self.args.dset.sigma_data, "unconditional", commit=False)

        return preds

    def setup_wandb_run(self, run):
        #get the wandb run object from outside (in trainer.py or somewhere else)
        self.wandb_run=run
        self.use_wandb=True

    def log_audio(self,preds, mode:str, commit=False):
        string=mode+"_"+self.args.tester.name
        audio_path=utils_logging.write_audio_file(preds,self.args.exp.sample_rate, string,path=self.args.model_dir)
        print(audio_path)
        self.wandb_run.log({"audio_"+str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},step=self.it, commit=False)
        #TODO: log spectrogram of the audio file to wandb
        spec_sample=utils_logging.plot_spectrogram_from_raw_audio(preds, self.args.logging.stft)

        self.wandb_run.log({"spec_"+str(string): spec_sample}, step=self.it, commit=commit)

    def test(self):
        #self.setup_wandb()
        for m in self.args.tester.modes:
            if m=="single_recording":
                self.evaluate_single_recording()
            if m=="unconditional":
                self.sample_unconditional()
            else:
                raise NotImplementedError
        self.it+=1

