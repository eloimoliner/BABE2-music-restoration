from datetime import date
import pandas as pd
import pickle
import torch
import torchaudio
import numpy as np
import utils.dnnlib as dnnlib
import os

import utils.logging as utils_logging
import wandb
import copy

import utils.blind_bwe_utils as blind_bwe_utils

import soundfile as sf

class Evaluator():
    def __init__(
        self, args=None, network=None, diff_params=None, test_set=None, device=None, it=None
    ):
        self.args=args
        self.network=torch.compile(network)
        #self.network=network
        #prnt number of parameters
        

        self.diff_params=copy.copy(diff_params)
        self.device=device
        #choose gpu as the device if possible
        if self.device is None:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network=network

        torch.backends.cudnn.benchmark = True

        today=date.today() 
        if it is None:
            self.it=0

        mode='test' #this is hardcoded for now, I'll have to figure out how to deal with the subdirectories once I want to test conditional sampling
        self.path_sampling=os.path.join(args.model_dir,mode+today.strftime("%d_%m_%Y")+"_"+str(self.it))
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

        #I have to rethink if I want to create the same sampler object to do conditional and unconditional sampling
        self.setup_sampler()

        self.use_wandb=False #hardcoded for now

    def setup_sampler(self):
        self.sampler=dnnlib.call_func_by_name(func_name=self.args.tester.sampler_callable, model=self.network,  diff_params=self.diff_params, args=self.args, rid=True) #rid is used to log some extra information

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        if self.args.exp.exp_name=="diffwave-sr":
            print(state_dict.keys())
            print("noise_schedukar",state_dict["noise_scheduler"])
            self.network.load_state_dict(state_dict['ema_model'])
            self.network.eval()
            print("ckpt loaded")
        else:
            try:
                print("load try 1")
                self.network.load_state_dict(state_dict['ema'])
            except:
                #self.network.load_state_dict(state_dict['model'])
                try:
                    print("load try 2")
                    dic_ema = {}
                    for (key, tensor) in zip(state_dict['model'].keys(), state_dict['ema_weights']):
                        dic_ema[key] = tensor
                    self.network.load_state_dict(dic_ema)
                except:
                    print("load try 3")
                    dic_ema = {}
                    i=0
                    for (key, tensor) in zip(state_dict['model'].keys(), state_dict['model'].values()):
                        if tensor.requires_grad:
                            dic_ema[key]=state_dict['ema_weights'][i]
                            i=i+1
                        else:
                            dic_ema[key]=tensor     
                    self.network.load_state_dict(dic_ema)
        try:
            self.it=state_dict['it']
        except:
            self.it=0

 
        if self.args.tester.evaluation.LTAS_file=="in_checkpoint":
            print(state_dict.keys())
            print(state_dict['LTAS'])
            self.LTAS_ref=state_dict['LTAS'].to(self.device)

    def test_real_blind_bwe_complete(self, typefilter="fc_A", compute_sweep=False, use_denoiser=True):
        #raise NotImplementedError

        #columns=["id", "original_audio", "degraded_audio", "reconstructed_audio", "degraded_estimate_audio", "estimate_filter"] 
        
        #res=torch.zeros((len(self.test_set),self.args.exp.audio_len))
        #the conditional sampling uses batch_size=1, so we need to loop over the test set. This is done for simplicity, but it is not the most efficient way to do it.
        filename=self.args.tester.complete_recording.path
        path, basename=os.path.split(filename)
        print(path, basename)

        d,fs=sf.read(filename)
        #stereo to mono
        print("d.shape",d.shape)
        if len(d.shape)>1:
            d=d.mean(axis=1)


        degraded=torch.Tensor(d)

        segL=self.args.exp.audio_len

        ix_first=self.args.exp.sample_rate*self.args.tester.complete_recording.ix_start #index of the first segment to be processed, might have to depend on the sample rate

        #for i, (degraded,  filename) in enumerate(tqdm(zip(test_set_data,  test_set_names))):

        print("filename",filename)
        n=os.path.splitext(os.path.basename(filename))[0]+typefilter
        degraded=degraded.float().to(self.device).unsqueeze(0)
        print(n)

        if self.args.tester.complete_recording.use_denoiser:
                print("denoising")
                if fs!=self.args.tester.denoiser.sample_rate_denoiser:
                    degraded=torchaudio.functional.resample(degraded, fs, self.args.tester.denoiser.sample_rate_denoiser)
                degraded=self.apply_denoiser(degraded)
                if fs!=self.args.tester.denoiser.sample_rate_denoiser:
                    degraded=torchaudio.functional.resample(degraded,  self.args.tester.denoiser.sample_rate_denoiser, self.args.exp.sample_rate)

                path_reconstructed=utils_logging.write_audio_file(degraded, self.args.exp.sample_rate, basename+".denoised.wav", path=path, normalize=True)
        else:
                print("no denoising")
                degraded=torchaudio.functional.resample(degraded, fs, self.args.exp.sample_rate)

        print("seg shape",degraded.shape)

        std= degraded.std(-1)
        degraded=self.args.tester.complete_recording.std*degraded/std.unsqueeze(-1)
        #add noise
        if self.args.tester.complete_recording.SNR_extra_noise!="None":
            #contaminate a bit with white noise
            SNR=10**(self.args.tester.complete_recording.SNR_extra_noise/10)
            sigma2_s=torch.Tensor([self.args.tester.complete_recording.std**2]).to(degraded.device)
            sigma=torch.sqrt(sigma2_s/SNR)
            degraded+=sigma*torch.randn(degraded.shape).to(degraded.device)


        if self.args.tester.complete_recording.n_segments_blindstep==1:
            y=degraded[...,ix_first:ix_first+segL]
        else:
            #initialize y with the first segment and repeat it
            y=degraded[...,ix_first:ix_first+segL].repeat(self.args.tester.complete_recording.n_segments_blindstep,1)
            for j in range(0, self.args.tester.complete_recording.n_segments_blindstep):
                #random index
                ix=np.random.randint(0, degraded.shape[-1]-segL)
                y[j,...]=degraded[...,ix:ix+segL]
        
        print("y shape",y.shape)

            

        #scale=self.LTAS_processor.rescale_audio_to_LTAS(y, fs)
        #print("scale",scale) #TODO I should calculate this with the whole track, not just the first segment

        #y=y*10**(scale/20)
        #degraded=degraded*10**(scale/20)



        rid=False
        outputs=self.sampler.predict_blind_bwe(y, rid=rid)
        pred, estimated_filter =outputs

        #now I will just throw away the first segment and process the rest of the signal with the estimated filter. Later I should think of a better way to do it

        overlap=int(self.args.tester.complete_recording.overlap*self.args.exp.sample_rate)
        hop=segL-overlap

        final_pred=torch.zeros_like(degraded)
        final_pred[0, ix_first:ix_first+segL]=pred[0]

        path_reconstructed=utils_logging.write_audio_file(final_pred, self.args.exp.sample_rate, basename+".reconstructed.wav", path=path, normalize=True)

        L=degraded.shape[-1]

        #modify the sampler, so that it is computationally cheaper

        discard_end=200 #discard the last 50 samples of the segment, because they are not used for the prediction
        discard_start=0  #discard the first 50 samples of the segment, because they are not used for the prediction

        #first segment
        ix=0
        seg=degraded[...,ix:ix+segL]
        pred=self.sampler.predict_bwe(seg, estimated_filter, typefilter,rid=False, test_filter_fit=False, compute_sweep=False)

        previous_pred=pred[..., 0:segL-discard_end]

        final_pred[...,ix:ix+segL-discard_end]=previous_pred
        ix+=segL-overlap-discard_end

        y_masked=torch.zeros_like(pred, device=self.device)
        mask=torch.ones_like(seg, device=self.device)
        mask[...,overlap::]=0

        hann_window=torch.hann_window(overlap*2, device=self.device)

        while ix<L-segL-discard_end-discard_start:
            y_masked[...,0:overlap]=previous_pred[...,segL-overlap-discard_end:]
            seg=degraded[...,ix:ix+segL]

            pred=self.sampler.predict_bwe_AR(seg, y_masked, estimated_filter, typefilter,rid=False, test_filter_fit=False, compute_sweep=False, mask=mask)

            previous_pred=pred[..., 0:segL-discard_end]


            final_pred[...,ix:ix+segL-discard_end]=previous_pred
            #do a little bit of overlap and add with a hann window to avoid discontinuities
            #final_pred[...,ix:ix+overlap]=final_pred[...,ix:ix+overlap]*hann_window[overlap::]+pred[...,0:overlap]*hann_window[0:overlap]
            #final_pred[...,ix+overlap:ix+segL]=pred[...,overlap::]

            path, basename=os.path.split(filename)
            print(path, basename)


            path_reconstructed=utils_logging.write_audio_file(final_pred, self.args.exp.sample_rate, basename+".reconstructed.wav", path=path, normalize=True)


            ix+=segL-overlap-discard_end

        #skipping the last segment, which is not complete, I am lazy
        seg=degraded[...,ix::]
        y_masked[...,0:overlap]=pred[...,-overlap::]

        if seg.shape[-1]<segL:
            #cat zeros
            seg_zp=torch.cat((seg, torch.zeros((1,segL-seg.shape[-1]), device=self.device)), -1)

            #the cat zeroes will also be part of the observed signal, so I need to mask them
            y_masked[...,seg.shape[-1]:segL]=seg_zp[...,seg.shape[-1]:segL]
            mask[...,seg.shape[-1]:segL]=0

        else:
            seg_zp=seg[...,0:segL]


        pred=self.sampler.predict_bwe_AR(seg_zp,y_masked, estimated_filter, typefilter,rid=False, test_filter_fit=False, compute_sweep=False, mask=mask)

        final_pred[...,ix::]=pred[...,0:seg.shape[-1]]

        final_pred=final_pred*std.unsqueeze(-1)/self.args.tester.complete_recording.std
        #final_pred=final_pred*10**(-scale/20)

        #extract path from filename


        path_reconstructed=utils_logging.write_audio_file(final_pred, self.args.exp.sample_rate, basename+".reconstructed.wav", path=path)
               
    def evaluate_multiple_recordings(self, process_complete=False, tworounds=False):

        path_dataset=self.args.tester.evaluation.path_dataset
        singer=self.args.tester.evaluation.singer
        csv_file=os.path.join(path_dataset, singer,self.args.tester.evaluation.csv_file)
        #read the csv file
        list_recordings=pd.read_csv(csv_file, header=None)
        for i, row in list_recordings.iterrows():
            recording=row[0]
            recording_path=os.path.join(path_dataset, singer, recording)
            path, basename=os.path.split(recording_path)
            if os.path.exists(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav")):
                continue
            self.evaluate_single_recording(recording_path=recording_path, process_complete=process_complete, tworounds=tworounds)
               
    def evaluate_single_recording(self, recording_path=None, process_complete=False,tworounds=False):

        #assert process_complete==False, "not implemented yet"

        if recording_path is None:
            path_dataset=self.args.tester.evaluation.path_dataset
            singer=self.args.tester.evaluation.singer
            recording=self.args.tester.evaluation.single_recording
            recording_path=os.path.join(path_dataset, singer, recording)

            #assert recording_path points to a  .wav file
            try: 
                if not os.path.exists(recording_path):
                    print("recording does not exist")
                    raise ValueError
            except:
                recording_path=recording
                if not os.path.exists(recording_path):
                    print("recording does not exist")
                    raise ValueError
        print("recording path",recording_path)

        #test_set_data=[]
        #test_set_fs=[]
        #test_set_names=[]
        recording,fs=sf.read(recording_path)
        print("recording",recording.shape)
        #stereo to mono
        if len(recording.shape)>1:
            recording=recording.mean(axis=1)
        assert fs==self.args.exp.sample_rate, "sample rate of the recording does not match the sample rate of the model"
        recording=torch.Tensor(recording)

        print("filename",recording_path)
        #segment the recording
        seg_mode=self.args.tester.evaluation.segment_placement
        seg_len=self.args.tester.evaluation.segment_length
        if seg_len != self.args.exp.audio_len:
            #necessary because of the CQT
            self.sampler.model.setup_seg_len(seg_len)

        print("num batch", self.args.tester.evaluation.num_segments_batch)
        num_segments_batch=self.args.tester.evaluation.num_segments_batch

        if self.args.tester.evaluation.LTAS_init:
            if self.args.tester.evaluation.LTAS_file=="in_checkpoint":
                LTAS_ref=self.LTAS_ref
            else:
                LTAS_ref=torch.load(self.args.tester.evaluation.LTAS_file)
 

        if not process_complete:
            if seg_mode=="random":
                #randomly select n segments (uniformly distributed)
                y=torch.zeros((num_segments_batch, seg_len))
                for i in range(0, num_segments_batch):
                    ix=np.random.randint(0, recording.shape[-1]-seg_len)
                    y[i,...]=torch.Tensor(recording[ix:ix+seg_len]).to(torch.float32)
            elif seg_mode=="middle":
                #select n segments from the middle of the recording. if there is more than one segment, then divide the recording in n parts of equal length, and select the middle segment of each part
                n_parts=num_segments_batch
                part_len=int(recording.shape[-1]/n_parts)
                y=torch.zeros((num_segments_batch, seg_len))
                for i in range(0, num_segments_batch):
                    ix=part_len*i+int(part_len/2)
                    y[i,...]=torch.Tensor(recording[ix:ix+seg_len]).to(torch.float32)
            elif seg_mode=="start_s":
                #the starts are defined in the config file
                starts=self.args.tester.evaluation.start_s #in seconds
                assert len(starts)==num_segments_batch, "number of starts does not match the number of segments"
                y=torch.zeros((num_segments_batch, seg_len))
                for i in range(0, num_segments_batch):
                    ix=int(starts[i]*fs)
                    assert ix+seg_len<recording.shape[-1], "segment is too long (or the start is too late)"
                    y[i,...]=torch.Tensor(recording[ix:ix+seg_len]).to(torch.float32)
            std= y.std() #a scalar
            print("std",std, "shape", y.shape)
            #normalize
            y=self.args.tester.blind_bwe.sigma_norm*y/std
        elif process_complete:
            if self.args.tester.evaluation.LTAS_init:
                nfft=4096
                freqs = np.fft.fftfreq(nfft, 1/self.args.exp.sample_rate)[0:nfft//2+1]
                freqs[-1]=self.args.exp.sample_rate/2
                freqs=torch.tensor(freqs, dtype=torch.float32).to(self.device)

                LTAS_ref=LTAS_ref.to(self.device)
                LTAS_smooth=blind_bwe_utils.smooth_LTAS(LTAS_ref,freqs,Noct=3)

                #max=LTAS_smooth.max()
                #correction=1/max #that correction may not be necessary, but I will keep it for now
                #LTAS_smooth=LTAS_smooth*correction


                LTAS_y, std=blind_bwe_utils.compute_LTAS(x=recording.clone(), sample_rate=self.args.exp.sample_rate, nfft=nfft, hop_length=nfft//4, win_length=nfft, normalize=1 if self.args.tester.evaluation.LTAS_file=="in_checkpoint" else self.args.dset.sigma_data )
                LTAS_y=LTAS_y.to(self.device)
                LTAS_y_smooth=blind_bwe_utils.smooth_LTAS(LTAS_y, freqs, Noct=3)

                diff=torch.clamp(10*torch.log10(LTAS_y_smooth) -10*torch.log10(LTAS_smooth),min=-20)

                #sorry for the hardcoding
                win_length = nfft//2
                hop_length = nfft//4

                X=torch.stft(recording.clone().to(self.device), n_fft=nfft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length).to(self.device), return_complex=True)
                #interpolate diff_lin to nfft/2+1

                #diff_lin_interp=torch.nn.functional.interpolate(diff_lin.unsqueeze(0).unsqueeze(0), size=nfft//2+1, mode='linear').squeeze(0).squeeze(0)
                diff_lin=torch.pow(10, diff/20)
                X/=diff_lin.unsqueeze(-1)

                x_init=torch.istft(X, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length).to(self.device), length=recording.shape[0])

                x_init=std*x_init/x_init.std()

                if self.args.tester.evaluation.LTAS_as_y:
                    recording=x_init


            if self.args.tester.evaluation.process_complete_mode=="AR_blind_and_complete":
                #divide the recording in segments of length seg_len (with overlap)
                overlap=int(self.args.tester.evaluation.overlap*seg_len)

                def segment_signal(recording, seg_len, overlap):
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

                segs, ix_start, ix_end,  expanded_size=segment_signal(recording, seg_len, overlap)
                result=torch.zeros((expanded_size,), device="cpu")
                result_mask=torch.zeros((expanded_size,), device="cpu")
                #calculate std of the recording, ignoring the silent passages
                std_s=[segs[i].std() for i in range(0, len(segs))]
                #take the median std
                std_orig=torch.median(torch.Tensor(std_s))
                segs=[self.args.tester.blind_bwe.sigma_norm*segs[i]/std_orig for i in range(0, len(segs))]

                processed_segs=[0 for i in range(0, len(segs))]

                #if seg_mode=="0":
                #    #select n segments from the middle of the recording. if there is more than one segment, then divide the recording in n parts of equal length, and select the middle segment of each part
                y=torch.zeros((1, seg_len))
                    
                y[0]=segs[0].to(torch.float32)

                if self.args.tester.evaluation.LTAS_init:

                    assert x_init.shape==recording.shape, "x_init and recording have different shapes: "+str(x_init.shape)+" and "+str(recording.shape)
                    segs_init, ix_start, ix_end,  expanded_size=segment_signal(x_init, seg_len, overlap)

                    #take the median std
                    std_orig=torch.median(torch.Tensor(std_s))
    
                    segs_init=[self.args.tester.blind_bwe.sigma_norm*segs_init[i]/std_orig for i in range(0, len(segs))]
    
                    x_seg_init=torch.zeros((1, seg_len))
                        
                    x_seg_init[0]=segs_init[0].to(torch.float32)

                    path, basename=os.path.split(recording_path)
                    sf.write(os.path.join(path, basename+"LTAS.wav"), x_init.cpu().cpu(), self.args.exp.sample_rate)

                #ix_start_blind=[]
                #ix_end_blind=[]
                #for i in range(0, num_segments_batch):
                #        ix=int(len(segs)/num_segments_batch*i+len(segs)/num_segments_batch/2)
                #        y[i,...]=segs[ix].to(torch.float32)
                #        processed_segs[ix]=1
                #        ix_start_blind.append(ix_start[ix])
                #        ix_end_blind.append(ix_end[ix])
            elif self.args.tester.evaluation.process_complete_mode=="LTI_blind_and_complete":
                #divide the recording in segments of length seg_len (with overlap)
                overlap=int(self.args.tester.evaluation.overlap*seg_len)


                def segment_signal(recording, seg_len, overlap):
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

                segs, ix_start, ix_end,  expanded_size=segment_signal(recording, seg_len, overlap)
                result=torch.zeros((expanded_size,), device="cpu")
                result_mask=torch.zeros((expanded_size,), device="cpu")
                #calculate std of the recording, ignoring the silent passages
                std_s=[segs[i].std() for i in range(0, len(segs))]
                #take the median std
                std_orig=torch.median(torch.Tensor(std_s))
                segs=[self.args.tester.blind_bwe.sigma_norm*segs[i]/std_orig for i in range(0, len(segs))]

                processed_segs=[0 for i in range(0, len(segs))]

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

                if self.args.tester.evaluation.LTAS_init:

                    assert x_init.shape==recording.shape, "x_init and recording have different shapes: "+str(x_init.shape)+" and "+str(recording.shape)
                    segs_init, ix_start, ix_end,  expanded_size=segment_signal(x_init, seg_len, overlap)

                    #take the median std
                    std_orig=torch.median(torch.Tensor(std_s))
    
                    segs_init=[self.args.tester.blind_bwe.sigma_norm*segs_init[i]/std_orig for i in range(0, len(segs))]
    
                    x_seg_init=torch.zeros((num_segments_batch, seg_len))
                    for i in range(0, num_segments_batch):
                        ix=int(len(segs)/num_segments_batch*i+len(segs)/num_segments_batch/2)
                        x_seg_init[i]=segs_init[ix].to(torch.float32)

                    
        path, basename=os.path.split(recording_path)
        if not process_complete:
            y=y.to(self.device)

            #seg=torchaudio.functional.resample(seg, fs, self.args.exp.sample_rate)

            rid=False
            outputs=self.sampler.predict_blind_bwe(y, rid=rid)
        
            pred, estimated_filter =outputs

        #save estimated_filter as a pkl file




            with open(os.path.join(path, basename+".filter_"+str(self.args.id)+".pkl"), "wb") as f:
                pickle.dump(estimated_filter, f)
        #save the results
            #reshape pred to be a 1d tensor ("now was shape (4, seg_len)")
            pred=pred.reshape(-1)
            sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), self.args.dset.sigma_data*pred.cpu().numpy(), self.args.exp.sample_rate)
        
                
        if process_complete:
            y=y.to(self.device)

            #seg=torchaudio.functional.resample(seg, fs, self.args.exp.sample_rate)

            rid=False
            print("std_orig",std_orig, y.std())
            #print("std_orig",std_orig, y.std())
            ##print("std_orig",std_orig, y.std())
            #print("std_orig",std_orig, y.std())
            if self.args.tester.evaluation.LTAS_init:
                x_init=x_seg_init
                print("x_init shape", x_init.shape)
            else:
                x_init=None

            print("y shape", y.shape)
            try:            
                outputs=self.sampler.predict_blind_bwe(y, rid=rid, x_init=x_init )
            except:            
                outputs=self.sampler.predict_blind_bwe(y, rid=rid )
        
            pred, estimated_filter =outputs
            print("pred", pred.shape)

            #process the complete recording with the estimated filter
            if self.args.tester.evaluation.process_complete_mode=="AR_blind_and_complete":
                y=y.to(self.device)

                rid=False
                outputs=self.sampler.predict_blind_bwe(y, rid=rid)
        
                pred, estimated_filter =outputs
                #add estimated_filter into a list
                estimated_filters=[estimated_filter]
                
                print("pred", pred[0].std())


                for i in range(0, pred.shape[0]):
                    result[ix_start[0]:ix_end[0]]=pred[i].cpu()
                    result_mask[ix_start[0]:ix_end[0]]=1
                #save result    
                
                sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), std_orig*result.cpu().numpy()/self.args.tester.blind_bwe.sigma_norm,  self.args.exp.sample_rate)

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
                        try:
                            out=self.sampler.predict_blind_bwe_AR(y, y_masked,rid=rid,  test_filter_fit=False, compute_sweep=False, mask=mask, x_init=x_seg_init if self.args.tester.evaluation.LTAS_init else None)
                        except:
                            out=self.sampler.predict_blind_bwe_AR(y, y_masked,rid=rid,  test_filter_fit=False, compute_sweep=False, mask=mask, x_init=x_seg_init if self.args.tester.evaluation.LTAS_init else None)
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
            elif self.args.tester.evaluation.process_complete_mode=="LTI_blind_and_complete":
                for i in range(0, pred.shape[0]):
                    result[ix_start_blind[i]:ix_end_blind[i]]=pred[i].cpu()
                    result_mask[ix_start_blind[i]:ix_end_blind[i]]=1
                #save result    
                sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), std_orig*result.cpu().numpy()/self.args.tester.blind_bwe.sigma_norm, self.args.exp.sample_rate)

                for i in range(0, len(segs)):
                    #process segs one by one
                    if processed_segs[i]==0:
                        y=segs[i].to(self.device)
                        y_masked=result[ix_start[i]:ix_end[i]].clone()
                        y_masked=y_masked.to(self.device)
                        mask=result_mask[ix_start[i]:ix_end[i]].clone()
                        mask=mask.to(self.device)
                        rid=False

                        if self.args.tester.evaluation.LTAS_init:
                            x_seg_init=segs_init[i].to(self.device)
                        else:
                            x_seg_init=None

                        out=self.sampler.predict_bwe_AR(y, y_masked, estimated_filter, "BABE2",rid=rid,  test_filter_fit=False, compute_sweep=False, mask=mask, x_init=x_seg_init)
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
                        sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), std_orig*result.cpu().numpy()/self.args.tester.blind_bwe.sigma_norm,  self.args.exp.sample_rate)

                result=result[0:recording.shape[-1]]
                sf.write(os.path.join(path, basename+".reconstructed_"+str(self.args.id)+".wav"), self.args.dset.sigma_data*result.cpu().numpy(), self.args.exp.sample_rate)



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
            if m=="single_example":
                self.evaluate_single_recording(process_complete=False)
            if m=="multiple_examples":
                self.evaluate_multiple_recordings(process_complete=False)
            if m=="single_recording":
                print("single recording")
                self.evaluate_single_recording(process_complete=True)
            if m=="multiple_recordings":
                self.evaluate_multiple_recordings(process_complete=True)
            if m=="multiple_recodings_FAD":
                self.evaluate_multiple_recodings()
                self.compute_FAD()
            if m=="multiple_recodings_pitch":
                self.evaluate_multiple_recodings()
                self.compute_pitch()
            if m=="multiple_recodings_pitch_FAD":
                self.evaluate_multiple_recodings()
                self.compute_FAD()
                self.compute_pitch()
        self.it+=1
