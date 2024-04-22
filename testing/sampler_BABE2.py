import os
from tqdm import tqdm
import torch
import torchaudio

import utils.blind_bwe_utils as blind_bwe_utils
import wandb
import omegaconf
import utils.logging as utils_logging


class LPFOperator():

    def __init__(self, args, device) -> None:
        self.args=args
        self.device=device

        self.fcmin=self.args.tester.blind_bwe.fcmin
        if self.args.tester.blind_bwe.fcmax =="nyquist":
                self.fcmax=self.args.exp.sample_rate//2
        else:
                self.fcmax=self.args.tester.blind_bwe.fcmax
        self.Amin=self.args.tester.blind_bwe.Amin
        self.Amax=self.args.tester.blind_bwe.Amax

        if self.args.tester.blind_bwe.optimization.last_slope_fixed:
            self.args.tester.blind_bwe.initial_conditions.A_p[-1]=self.Amin
        if self.args.tester.blind_bwe.optimization.first_slope_fixed:
            self.args.tester.blind_bwe.initial_conditions.A_m[-1]=self.Amax

        self.params_fref=torch.Tensor([args.tester.blind_bwe.initial_conditions.fref]).to(device)
        self.params_fc_p=torch.Tensor(self.args.tester.blind_bwe.initial_conditions.fc_p).to(device)
        assert (self.params_fc_p > self.params_fref).all(), "fc_p must be greater than fref"
        self.params_fc_p=torch.nn.Parameter(self.params_fc_p)

        self.params_fc_m=torch.Tensor(self.args.tester.blind_bwe.initial_conditions.fc_m).to(device)
        assert (self.params_fc_m < self.params_fref).all(), "fc_m must be smaller than fref"
        self.params_fc_m=torch.nn.Parameter(self.params_fc_m)

        self.params_A_p=torch.Tensor(self.args.tester.blind_bwe.initial_conditions.A_p).to(device)
        self.params_A_p=torch.nn.Parameter(self.params_A_p)
        self.params_A_m=torch.Tensor(self.args.tester.blind_bwe.initial_conditions.A_m).to(device)
        self.params_A_m=torch.nn.Parameter(self.params_A_m)


        self.freqs=torch.fft.rfftfreq(self.args.tester.blind_bwe.NFFT, d=1/self.args.exp.sample_rate).to(self.device) #

        self.params=[self.params_fref,self.params_fc_p, self.params_fc_m, self.params_A_p, self.params_A_m]

        self.optimizer=torch.optim.Adam(self.params, lr=self.args.tester.blind_bwe.lr_filter) #self.mu=torch.Tensor([self.args.tester.blind_bwe.optimization.mu[0], self.args.tester.blind_bwe.optimization.mu[1]])

        self.tol=self.args.tester.blind_bwe.optimization.tol

    def assign_params(self, params):
        assert len(params[1])==(len(params[3])-1)
        assert len(params[2])==(len(params[4])-1)
        self.params_fref=params[0]
        self.params_fc_p=params[1]
        self.params_fc_m=params[2]
        self.params_A_p=params[3]
        self.params_A_m=params[4]
        self.params=[self.params_fref,self.params_fc_p, self.params_fc_m, self.params_A_p, self.params_A_m]

    def degradation(self, x):
        return self.apply_filter_fcA(x)

    def apply_filter_fcA(self, x):
        H=blind_bwe_utils.design_filter_3(self.params, self.freqs, block_low_freq=self.args.tester.blind_bwe.optimization.block_low_freq)
        return blind_bwe_utils.apply_filter(x, H,self.args.tester.blind_bwe.NFFT)

    def stop(self, prev_params):

        #raise NotImplementedError
        decision=False
        if (torch.abs(self.params[0]-prev_params[0]).mean()<self.tol[0]):
            if (torch.abs(self.params[1]-prev_params[1]).mean()<self.tol[0]):
                if (torch.abs(self.params[2]-prev_params[2]).mean()<self.tol[0]):
                    if (torch.abs(self.params[3]-prev_params[3]).mean()<self.tol[1]):
                        if (torch.abs(self.params[4]-prev_params[4]).mean()<self.tol[1]):
                            print("tolerance reached")
                            decision=True
        return decision
    def collapse_regularization(self):
        dist=[]

        #fc_p[0]-fref
        dist.append(self.params[1][0]-self.params[0][0])

        for i in range(1,len(self.params[1])):
            dist.append(self.params[1][i]-self.params[1][i-1])
        dist.append(self.params[1][-1]-self.fcmax)
        
        #fc_m[0]-fref
        dist.append(self.params[0][0]-self.params[2][0])
        for i in range(1,len(self.params[2])):
            dist.append(self.params[2][i]-self.params[2][i-1])

        dist.append(self.fcmin-self.params[2][-1])

        beta=self.args.tester.collapse_regularization.beta
        gamma=self.args.tester.collapse_regularization.gamma
        cost=[torch.exp(-beta*x.abs()**gamma) for x in dist]
        return torch.stack(cost).sum()
    
    def generate_guidance_labels(self, clean_audio):
    # Convert the clean audio to a one-hot representation
        clean_audio_one_hot = torch.nn.functional.one_hot(clean_audio).float().to(self.device)
        return clean_audio_one_hot
    def limit_params(self):

        
        for i in range(len(self.params)):
            self.params[i].detach_()

        self.params[0][0]=torch.clamp(self.params[0][0],min=self.fcmin,max=self.fcmax)
        if self.args.tester.blind_bwe.optimization.clamp_fc:
                #"fc_p"
                self.params[1][0]=torch.clamp(self.params[1][0],min=self.params[0][0]+1e-3,max=self.fcmax)
                for k in range(1,len(self.params[1])):
                    self.params[1][k]=torch.clamp(self.params[1][k],min=self.params[1][k-1]+1e-3,max=self.fcmax)

                #"fc_m"
                self.params[2][0]=torch.clamp(self.params[2][0],min=self.fcmin,max=self.params[0][0]-1e-3)
                for k in range(1,len(self.params[2])):
                    self.params[2][k]=torch.clamp(self.params[2][k],min=self.fcmin,max=self.params[2][k-1]-1e-3)

        assert (self.params[1]<=self.params[0][0]).any()==False, f"fc_p must be greater than fref: {self.params[1]}, {self.params[0][0]}"
        assert (self.params[2]>=self.params[0][0]).any()==False, f"fc_m must be smaller than fre: {self.params[2]}, {self.params[0][0]}"

        assert (self.params[2] <= self.freqs[1]).any()==False, f"fc_m must be greater than the minimum frequency: {self.params[2]}, {self.freqs[1]}"
        assert (self.params[1] >= self.freqs[-1]).any()==False, f"fc_p must be smaller than the maximum frequency: {self.params[1]}, {self.freqs[-1]}"


        if self.args.tester.blind_bwe.optimization.clamp_A:
            #"A_p"
            if self.args.tester.blind_bwe.optimization.only_negative_Ap:
                self.params[3][0]=torch.clamp(self.params[3][0],min=self.Amin,max=0)
                for k in range(1,len(self.params[3])):
                    self.params[3][k]=torch.clamp(self.params[3][k],min=self.Amin,max=self.params[3][k-1]-1e-1)
                    #self.params[2][k]=torch.clamp(self.params[2][k],min=self.Amin,max=0)
            else:
                for k in range(len(self.params[3])):
                    self.params[3][k]=torch.clamp(self.params[3][k],min=self.Amin,max=self.Amax)
            
            #"A_m"
            for k in range(len(self.params[4])):
                self.params[4][k]=torch.clamp(self.params[4][k],min=self.Amin,max=self.Amax)
            #self.params[1,0]=torch.clamp(self.params[1,0],min=self.Amin,max=-1 if self.args.tester.blind_bwe.optimization.only_negative_A else self.Amax)
        if self.args.tester.blind_bwe.optimization.last_slope_fixed:
            self.params[3][-1]=-self.args.tester.blind_bwe.Alim
        if self.args.tester.blind_bwe.optimization.first_slope_fixed:
            self.params[4][-1]=self.args.tester.blind_bwe.Alim

    def optimizer_func(self, x_hat, y, clean_audio):
        """
        Xden: STFT of denoised estimate
        y: observations
        params: parameters of the degradation model (fc, A)
        """

        #if self.args.tester.blind_bwe.optimization.last_slope_fixed:
        #    self.params[3][-1]=self.Amin
        #if self.args.tester.blind_bwe.optimization.first_slope_fixed:
        #    self.params[4][-1]=self.Amax

        #print("before design filter", self.params)
        H = blind_bwe_utils.design_filter_3(self.params, self.freqs, block_low_freq=self.args.tester.blind_bwe.optimization.block_low_freq)
        return blind_bwe_utils.apply_filter_and_norm_STFTmag_fweighted(x_hat, y, H, self.args.tester.posterior_sampling.freq_weighting_filter, clean_audio)

    def apply_filter_and_norm_STFTmag_fweighted(x_hat, y, H, freq_weighting_filter, clean_audio):
        y_lpf = blind_bwe_utils.apply_filter(y, H, NFFT)
        x_lpf = blind_bwe_utils.apply_filter(x_hat, H, NFFT)

        # Generate guidance labels
        guidance_labels = generate_guidance_labels(clean_audio)

        # Calculate the CFG loss
        cfg_loss = torch.sum(guidance_labels * x_hat, dim=-1)
        cfg_loss = cfg_loss.mean()

    # Compute the total loss as the sum of reconstruction loss and CFG loss
        total_loss = norm_STFTmag_fweighted(x_lpf, y_lpf, freq_weighting_filter) + cfg_loss
        return total_loss
    def optimize_params(self, denoised_estimate, y):

        Xden=blind_bwe_utils.apply_stft(denoised_estimate, self.args.tester.blind_bwe.NFFT)
        Y=blind_bwe_utils.apply_stft(y, self.args.tester.blind_bwe.NFFT)

        #self.mu=self.mu.to(y.device)

        for i in tqdm(range(self.args.tester.blind_bwe.optimization.max_iter)):

            for j in range(len(self.params)):
                self.params[j].requires_grad=True
                #fc.requires_grad=True
            self.optimizer.zero_grad()

            loss=self.optimizer_func(Xden, Y)
            
            if self.args.tester.collapse_regularization.use:
                cost=self.collapse_regularization()
                loss+=self.args.tester.collapse_regularization.lambda_reg*cost


            loss.backward()

            #grad=torch.autograd.grad(norm,self.params,create_graph=True)
            #update params with gradient descent, using backtracking line search
            #print("before",self.params)
            #if i==0:
            #    for j in range(len(self.params)):
            #        print("grad",self.params[j].grad)
            self.optimizer.step()
            #print("update",self.params)

            #t=self.mu
            #newparams=self.params-t.unsqueeze(1)*grad[0]

            #update with the found step size
            #self.params=newparams

            #self.params[i].detach_()
            #limit params to help stability

            #with torch.no_grad():
            self.limit_params()


            #print("clamped",self.params)
    

            if i>0:
                if  self.stop(prev_params):
                    break

            prev_params=[self.params[i].clone().detach() for i in range(len(self.params))]

        print(self.params)

class AR_LPFOperator(LPFOperator):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.mask=None

    def degradation(self, x):
        return self.mask*x +(1-self.mask)*self.apply_filter_fcA(x)

class BlindSampler():

    def __init__(self, model,  diff_params, args):

        self.model = model

        self.diff_params = diff_params #same as training, useful if we need to apply a wrapper or something
        self.args=args
        if not(self.args.tester.diff_params.same_as_training):
            self.update_diff_params()

        self.order=self.args.tester.order

        self.xi=self.args.tester.posterior_sampling.xi
        #hyperparameter for the reconstruction guidance
        self.data_consistency=self.args.tester.posterior_sampling.data_consistency #use reconstruction gudance without replacement
        self.nb_steps=self.args.tester.T
        

        self.start_sigma=self.args.tester.posterior_sampling.start_sigma
        if self.start_sigma =="None":
            self.start_sigma=None

        print("start sigma", self.start_sigma)

        self.operator=None

        def loss_fn_rec(x_hat, x):
                diff=x_hat-x
                #if self.args.tester.filter_out_cqt_DC_Nyq:
                #    diff=self.model.CQTransform.apply_hpf_DC(diff)
                return (diff**2).sum()/2

        self.rec_distance=lambda x_hat, x: loss_fn_rec(x_hat, x)


    def update_diff_params(self):
        #the parameters for testing might not be necesarily the same as the ones used for training
        self.diff_params.sigma_min=self.args.tester.diff_params.sigma_min
        self.diff_params.sigma_max =self.args.tester.diff_params.sigma_max
        self.diff_params.ro=self.args.tester.diff_params.ro
        self.diff_params.sigma_data=self.args.tester.diff_params.sigma_data
        self.diff_params.Schurn=self.args.tester.diff_params.Schurn
        self.diff_params.Stmin=self.args.tester.diff_params.Stmin
        self.diff_params.Stmax=self.args.tester.diff_params.Stmax
        self.diff_params.Snoise=self.args.tester.diff_params.Snoise


    def get_rec_grads(self, x_hat, y, x, t_i, clean_audio):
        """
        Compute the gradients of the reconstruction error with respect to the input
        """ 

        if self.args.tester.posterior_sampling.annealing_y.use:
            if self.args.tester.posterior_sampling.annealing_y.mode=="same_as_x":
                y=y+torch.randn_like(y)*t_i
            elif self.args.tester.posterior_sampling.annealing_y.mode=="same_as_x_limited":
                t_min=torch.Tensor([self.args.tester.posterior_sampling.annealing_y.sigma_min]).to(y.device)
                #print(t_i, t_min)
                t_y=torch.max(t_i, t_min)
                #print(t_y)

                y=y+torch.randn_like(y)*t_y
            elif self.args.tester.posterior_sampling.annealing_y.mode=="fixed":
                t_min=torch.Tensor([self.args.tester.posterior_sampling.annealing_y.sigma_min]).to(y.device)
                #print(t_i, t_min)

                y=y+torch.randn_like(y)*t_min

        print("y",y.std(), "x_hat",x_hat.std())
        norm = self.rec_distance(self.operator.degradation(x_hat), y)
        print("norm:", norm.item())

        # Generate guidance labels
        guidance_labels = self.generate_guidance_labels(clean_audio)

         # Calculate the CFG loss
        cfg_loss = torch.sum(guidance_labels * x_hat, dim=-1)
        cfg_loss = cfg_loss.mean()
        
        # Compute the total loss as the sum of reconstruction loss and CFG loss
        total_loss = norm + self.args.tester.cfg.guidance_weight * cfg_loss

        rec_grads = torch.autograd.grad(outputs=total_loss.sum(),
                                    inputs=x)

        rec_grads = rec_grads[0]

        rec_grads=torch.autograd.grad(outputs=norm.sum(),
                                      inputs=x)

        rec_grads=rec_grads[0]
        
        if self.args.tester.posterior_sampling.normalization=="grad_norm":
        
            normguide=torch.norm(rec_grads)/self.args.exp.audio_len**0.5
            #normguide=norm/self.args.exp.audio_len**0.5
        
            #normalize scaling
            s=self.xi/(normguide+1e-6)

        elif self.args.tester.posterior_sampling.normalization=="loss_norm":
            normguide=norm/self.args.exp.audio_len**0.5
            #normguide=norm/self.args.exp.audio_len**0.5
        
            #normalize scaling
            s=self.xi/(normguide+1e-6)
        
        #optionally apply a treshold to the gradients
        if False:
            #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
            rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        
        return s*rec_grads/t_i, norm

    def get_score_rec_guidance(self, x, y, t_i):

        x.requires_grad_()
        x_hat=self.get_denoised_estimate(x, t_i)

        rec_grads=self.get_rec_grads(x_hat, y, x, t_i)

        
        score=self.denoised2score(x_hat, x, t_i)

        score=score-rec_grads

        return score
    
    def get_denoised_estimate(self, x, t_i):

        #assuming you have here some noisy signal y (take care of the message passing)

        x_hat=self.diff_params.denoiser(x,y, self.model, t_i.unsqueeze(-1))

        if self.args.tester.filter_out_cqt_DC_Nyq:
            x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)
        return x_hat
    

    def get_score(self,x, y, t_i):
        if y==None:
            assert self.operator==None
            #unconditional sampling
            with torch.no_grad():
                x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                if self.args.tester.filter_out_cqt_DC_Nyq:
                    x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)
                score=(x_hat-x)/t_i**2
            return score
        else:
            if self.xi>0:
                assert self.operator is not None
                #apply rec. guidance
                score=self.get_score_rec_guidance(x, y, t_i)
    
                #optionally apply replacement or consistency step
                if self.data_consistency:
                    raise NotImplementedError
                    #convert score to denoised estimate using Tweedie's formula
                    x_hat=score*t_i**2+x
    
                    try:
                        x_hat=self.data_consistency_step(x_hat)
                    except:
                        x_hat=self.data_consistency_step(x_hat,y, degradation)
    
                    #convert back to score
                    score=(x_hat-x)/t_i**2
    
            else:
                #raise NotImplementedError
                #denoised with replacement method
                with torch.no_grad():
                    x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                        
                    #x_hat=self.data_consistency_step(x_hat,y, degradation)
                    if self.data_consistency:
                        raise NotImplementedError
                        try:
                            x_hat=self.data_consistency_step(x_hat)
                        except:
                            try:
                                x_hat=self.data_consistency_step(x_hat,y, degradation)
                            except:
                                x_hat=self.data_consistency_step(x_hat,y, degradation, filter_params)

        
                    score=(x_hat-x)/t_i**2
    
            return score

    def apply_FIR_filter(self,y):
        y=y.unsqueeze(1)

        #apply the filter with a convolution (it is an FIR)
        y_lpf=torch.nn.functional.conv1d(y,self.filt,padding="same")
        y_lpf=y_lpf.squeeze(1) 

        return y_lpf
    def apply_IIR_filter(self,y):
        y_lpf=torchaudio.functional.lfilter(y, self.a,self.b, clamp=False)
        return y_lpf
    def apply_biquad(self,y):
        y_lpf=torchaudio.functional.biquad(y, self.b0, self.b1, self.b2, self.a0, self.a1, self.a2)
        return y_lpf
    def decimate(self,x):
        return x[...,0:-1:self.factor]

    def resample(self,x):
        N=100
        return torchaudio.functional.resample(x,orig_freq=int(N*self.factor), new_freq=N)

    #def apply_3rdoct_filt(self,x, filt, freq_octs):
    #    filt=f_utils.unnormalize_filter(filt)
    #    y=f_utils.apply_filter(x, filt, self.args.tester.blind_bwe.NFFT,self.args.exp.sample_rate, freq_octs.to(x.device),interpolation="hermite_cubic") 
    #    return y[0]
    def prepare_smooth_mask(self, mask, size=10):
        hann=torch.hann_window(size*2)
        hann_left=hann[0:size]
        hann_right=hann[size::]
        B,N=mask.shape
        mask=mask[0]
        prev=1
        new_mask=mask.clone()
        #print(hann.shape)
        for i in range(len(mask)):
            if mask[i] != prev:
                #print(i, mask.shape, mask[i], prev)
                #transition
                if mask[i]==0:
                   print("apply right")
                   #gap encountered, apply hann right before
                   new_mask[i-size:i]=hann_right
                if mask[i]==1:
                   print("apply left")
                   #gap encountered, apply hann left after
                   new_mask[i:i+size]=hann_left
                #print(mask[i-2*size:i+2*size])
                #print(new_mask[i-2*size:i+2*size])
                
            prev=mask[i]
        return new_mask.unsqueeze(0).expand(B,-1)

    def predict_bwe_AR(
        self,
        ylpf,  #observations (lowpssed signal) Tensor with shape (L,)
        y_masked,
        filt, #filter Tensor with shape ??
        mask=None,
        x_init=None
        ):
        assert mask is not None

        self.operator=AR_LPFOperator(self.args, ylpf.device)
        print("fc_A")

        #self.freqs=torch.fft.rfftfreq(self.args.tester.blind_bwe.NFFT, d=1/self.args.exp.sample_rate).to(ylpf.device)
        self.operator.assign_params(filt)
        self.operator.mask=mask
        y=mask*y_masked+(1-mask)*ylpf


        return self.predict_conditional(y,  x_init=x_init)

        
    def predict_bwe(
        self,
        ylpf,  #observations (lowpssed signal) Tensor with shape (L,)
        filt, #filter Tensor with shape ??
        reference=None
        ):

        self.reference=reference

        self.operator=LPFOperator(self.args, ylpf.device)
        print("fc_A")

        self.operator.assign_params(filt)
        

        return self.predict_conditional(ylpf)

    def predict_unconditional(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device,
    ):
        self.y=None
        self.degradation=None
        self.reference=None
        #self.args.tester.posterior_sampling.xi=0    #just in case
        #self.args.tester.posterior_sampling.start_sigma="None"    #just in case
        self.start_sigma=None
        self.xi=0
        return self.predict(shape=shape, device=device, conditional=False)



    def predict_conditional(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        x_init=None
    ):
        self.y=y
        return self.predict(shape=y.shape,device=y.device,  blind=False, x_init=x_init)

  
    def denoised2score(self,  x_d0, x, t):
        #tweedie's score function
        return (x_d0-x)/t**2
    def score2denoised(self, score, x, t):
        return score*t**2+x

    def move_timestep(self, x, t, gamma, Snoise=1):
        #if gamma_sig[i]==0 this is a deterministic step, make sure it doed not crash
        t_hat=t+gamma*t
        #sample noise, Snoise is 1 by default
        epsilon=torch.randn(x.shape).to(x.device)*Snoise
        #add extra noise
        x_hat=x+((t_hat**2 - t**2)**(1/2))*epsilon
        return x_hat, t_hat


    def fit_params(self, denoised_estimate, y):

        Xden=blind_bwe_utils.apply_stft(denoised_estimate, self.args.tester.blind_bwe.NFFT)
        Y=blind_bwe_utils.apply_stft(y, self.args.tester.blind_bwe.NFFT)

        for i in tqdm(range(self.args.tester.blind_bwe.optimization.max_iter)):

            for j in range(len(self.operator.params)):
                self.operator.params[j].requires_grad=True
                #fc.requires_grad=True
            self.operator.optimizer.zero_grad()

            rec_loss=self.operator.optimizer_func(Xden, Y)

            #loss=rec_loss
            if self.args.tester.collapse_regularization.use:
                cost=self.operator.collapse_regularization()
                loss=rec_loss+self.args.tester.collapse_regularization.lambda_reg*cost
            else:
                loss=rec_loss


            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.operator.params, self.args.tester.blind_bwe.optimization.grad_clip)

            self.operator.optimizer.step()
            self.operator.limit_params()


            if i>0:
                if  self.operator.stop(prev_params):
                    break

            prev_params=[self.operator.params[k].clone().detach() for k in range(len(self.operator.params))]

        print(self.operator.params)

        if self.args.tester.wandb.use:
            self.wandb_run.log({"rec_loss_operator": rec_loss.item()}, step=self.step_count)
            if self.args.tester.collapse_regularization.use:
                self.wandb_run.log({"collapse_cost": cost.item()}, step=self.step_count)


        #self.operator.optimize_params(denoised_estimate, y)

    def step(self, x, t_i, t_i_1, gamma_i, blind=False, y=None, clean_audio=None):

        if self.args.tester.posterior_sampling.SNR_observations !="None":
            snr=10**(self.args.tester.posterior_sampling.SNR_observations/10)
            sigma2_s=torch.var(y, -1) 
            sigma=torch.sqrt(sigma2_s/snr).unsqueeze(-1)
            #sigma=torch.tensor([self.args.tester.posterior_sampling.sigma_observations]).unsqueeze(-1).to(y.device)
            #print(y.shape, sigma.shape)
            y=y+sigma*torch.randn(y.shape).to(y.device)
            

        x_hat, t_hat=self.move_timestep(x, t_i, gamma_i, self.diff_params.Snoise)

        x_hat.requires_grad_(True)

        x_den=self.get_denoised_estimate(x_hat, t_hat)

        x_den_2=x_den.clone().detach()

        if blind:
            self.fit_params(x_den_2, y)

        if self.args.tester.posterior_sampling.xi>0 and y is not None:
            rec_grads, rec_loss=self.get_rec_grads(x_den, y, x_hat, t_hat, clean_audio)
        else:
            #adding this so that the code does not crash
            rec_loss=0
            rec_grads=0

        x_hat.detach_()
        uncond_score=self.denoised2score(x_den_2, x_hat, t_hat)

        score=uncond_score-rec_grads

        d=-t_hat*score
        #apply second order correction
        h=t_i_1-t_hat


        if t_i_1!=0 and self.order==2:  #always except last step
            #second order correction2
            #h=t[i+1]-t_hat
            t_prime=t_i_1
            x_prime=x_hat+h*d
            x_prime.requires_grad_(True)

            x_den=self.get_denoised_estimate(x_prime, t_prime)

            x_den_2=x_den.clone().detach()

            if blind:
                self.fit_params(x_den_2, y)

            if self.xi>0 and y is not None:
                rec_grads, rec_loss =self.get_rec_grads(x_den, y, x_prime, t_prime)
            else:
                rec_loss=0
                rec_grads=0

            x_prime.detach_()

            uncond_score=self.denoised2score(x_den_2, x_prime, t_prime)
            
            score=uncond_score-rec_grads


            d_prime=-t_prime*score

            x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

        elif self.order==1: #first condition  is to avoid dividing by 0
            #first order Euler step
            x=x_hat+h*d

        return x, x_den_2, rec_loss, score, rec_grads

    def predict_blind_bwe_AR(
        self,
        ylpf,  #observations (lowpssed signal) Tensor with shape (L,)
        y_masked,
        reference=None,
        mask=None,
        x_init=None
        ):

        self.operator=AR_LPFOperator(self.args, ylpf.device)

        self.operator.mask=mask
        y=mask*y_masked+(1-mask)*ylpf

        self.reference=reference

        y=y.unsqueeze(0)

        self.y=y
        return self.predict(shape=y.shape, device=y.device, blind=True, x_init=x_init, clean_audio=clean_audio)

    def predict_blind_bwe(
        self,
        y,  #observations (lowpssed signal) Tensor with shape (L,)
        reference=None,
        x_init=None,
        ):
        self.operator=LPFOperator(self.args, y.device)
        self.reference=reference
        self.y=y
        return self.predict(shape=y.shape, device=y.device, blind=True, x_init=x_init, clean_audio=clean_audio)


    def predict(
        self,
        shape=None,  #observations (lowpssed signal) Tensor with shape (L,)
        device=None,
        blind=False,
        x_init=None,
        conditional=True,
        clean_audio=None,
        ):
        if not conditional:
            self.y=None

        if self.args.tester.wandb.use:
            self.setup_wandb()


        #get shape and device from the observations tensor
        if shape is None:
            shape=self.y.shape
            device=self.y.device

        #initialization
        if self.start_sigma is None:
            t=self.diff_params.create_schedule(self.nb_steps).to(device)
            x=self.diff_params.sample_prior(shape, t[0]).to(device)
        else:
            #get the noise schedule
            t = self.diff_params.create_schedule_from_initial_t(self.start_sigma,self.nb_steps).to(self.y.device)
            #sample from gaussian distribution with sigma_max variance
            if x_init is not None:
                x = x_init.to(device) + self.diff_params.sample_prior(shape,t[0]).to(device)
            else:
                print("using y as warm init")
                x = self.y + self.diff_params.sample_prior(shape,t[0]).to(device)

        gamma=self.diff_params.get_gamma(t).to(device)

        if self.args.tester.wandb.use:
            self.wandb_run.log({"y": wandb.Audio(self.args.dset.sigma_data*self.y.squeeze().detach().cpu().numpy(), caption="y", sample_rate=self.args.exp.sample_rate)}, step=0, commit=False) 

            spec_sample=utils_logging.plot_spectrogram_from_raw_audio(self.args.dset.sigma_data*self.y, self.args.logging.stft)
            self.wandb_run.log({"spec_y": spec_sample}, step=0, commit=False)

        if self.reference is not None:
            if self.args.tester.wandb.use:
                self.wandb_run.log({"reference": wandb.Audio(self.args.dset.sigma_data*self.reference.squeeze().detach().cpu().numpy(), caption="reference", sample_rate=self.args.exp.sample_rate)}, step=0, commit=False)

                spec_sample=utils_logging.plot_spectrogram_from_raw_audio(self.args.dset.sigma_data*self.reference, self.args.logging.stft)
                self.wandb_run.log({"spec_reference": spec_sample}, step=0, commit=True)


        for i in tqdm(range(0, self.nb_steps, 1)):
            self.step_count=i

            out = self.step(x, t[i], t[i+1], gamma[i], blind=blind, y=self.y, clean_audio=clean_audio)
            x, x_den, rec_loss, score, lh_score =out
            if self.y is not None:
                score_norm=score.norm()
                lh_socore_norm=lh_score.norm()

            if self.args.tester.wandb.use:
                self.wandb_run.log({"rec_loss": rec_loss, "score_norm": score_norm, "lh_socore_norm": lh_socore_norm}, step=i, commit=False)

                if blind:
                    #Plot filter
                    fig=blind_bwe_utils.plot_single_filter_BABE2_3(self.operator.params, self.operator.freqs)
                    self.wandb_run.log({"filter": fig}, step=i, commit=False)

                if i%self.args.tester.wandb.heavy_log_interval==0:

                    #these logs are only interesting for blind optimization
                    if self.reference is not None:
                        with torch.no_grad():
                            y_est_reference=self.operator.degradation(self.reference)
                        self.wandb_run.log({"y_est_reference": wandb.Audio(self.args.dset.sigma_data*y_est_reference.squeeze().detach().cpu().numpy(), caption="y_est_reference", sample_rate=self.args.exp.sample_rate)}, step=i, commit=False)
                        spec_sample=utils_logging.plot_spectrogram_from_raw_audio(self.args.dset.sigma_data*y_est_reference, self.args.logging.stft)
                        self.wandb_run.log({"spec_y_est_reference": spec_sample}, step=i, commit=False)
    
                    with torch.no_grad():
                        y_est=self.operator.degradation(x_den)
                    self.wandb_run.log({"y_est": wandb.Audio(self.args.dset.sigma_data*y_est.squeeze().detach().cpu().numpy(), caption="y_est", sample_rate=self.args.exp.sample_rate)}, step=i, commit=False)
            
                    spec_sample=utils_logging.plot_spectrogram_from_raw_audio(self.args.dset.sigma_data*y_est, self.args.logging.stft)
                    self.wandb_run.log({"spec_y_est": spec_sample}, step=i, commit=False)
    
                    #log audio
                    self.wandb_run.log({"x_den": wandb.Audio(self.args.dset.sigma_data*x_den.squeeze().detach().cpu().numpy(), caption="x_den", sample_rate=self.args.exp.sample_rate)}, step=i, commit=False)
    
                    spec_sample=utils_logging.plot_spectrogram_from_raw_audio(self.args.dset.sigma_data*x_den.detach(), self.args.logging.stft)
                    self.wandb_run.log({"spec_x": spec_sample}, step=i, commit=True)
                    #x_den.requires_grad_(True)

        if self.args.tester.wandb.use:
            self.wandb_run.finish()

        if blind:
            return x.detach() , self.operator.params
        else:
            return x.detach()
        

    def setup_wandb(self):
        config=omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        self.wandb_run=wandb.init(project=self.args.tester.wandb.project, entity=self.args.tester.wandb.entity, config=config)
        if self.operator.__class__.__name__== "WaveNetOperator":
            wandb.watch(self.operator.wavenet, log_freq=self.args.tester.wandb.heavy_log_interval, log="all") #wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name=self.args.tester.wandb.run_name +os.path.basename(self.args.model_dir)+"_"+self.args.exp.exp_name+"_"+self.wandb_run.id
        #adding the experiment number to the run name, bery important, I hope this does not crash
        self.use_wandb=True

