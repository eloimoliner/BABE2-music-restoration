import os
import time
import copy
import numpy as np
import torch
from utils.torch_utils import training_stats
from glob import glob
import re
import wandb
import utils.logging as utils_logging
import omegaconf
import utils.training_utils as t_utils
import torch.nn as nn   
from tqdm import tqdm

#----------------------------------------------------------------------------

class RFF_MLP_Block(nn.Module):
    """
        Encoder of the noise level embedding
        Consists of:
            -Random Fourier Feature embedding
            -MLP
    """
    def __init__(self, rff_dim=32, scale=1, init=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(rff_dim) * scale, requires_grad=False)
        self.phase = nn.Parameter(torch.rand(rff_dim), requires_grad=False)
        #self.RFF_freq = nn.Parameter(
        #    16 * torch.randn([1, rff_dim]), requires_grad=False)
        self.linear=nn.Linear(2*rff_dim, 1, bias=False)
        #self.linear.weight.data.zero_()

    def forward(self, x):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: weight of sigma
              (shape: [B, 1], dtype: float32)
        """
        x_proj = (x[:, None] * self.weight[None, :] + self.phase[None, :]) * 2 * np.pi
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.linear(out)

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class Trainer():
    def __init__(self, args, dset, network, optimizer, diff_params, tester=None, device='cpu'):
        self.args=args
        self.dset=dset
        #self.network=torch.compile(network)
        self.network=network

        self.optimizer=optimizer
        self.diff_params=diff_params
        self.device=device

        #testing means generating demos by sampling from the model
        self.tester=tester
        if self.tester is None or not(self.args.tester.do_test):
            self.do_test=False
        else:
            self.do_test=True

        torch.manual_seed(np.random.randint(1 << 31))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False


        if self.args.exp.lossmlp.use:
            self.lossmlp=RFF_MLP_Block(rff_dim=self.args.exp.lossmlp.rff_dim).to(self.device)
            self.optimizer=torch.optim.Adam(list(self.network.parameters())+ list(self.lossmlp.parameters()), lr=self.args.exp.lr, betas=(self.args.exp.optimizer.beta1, self.args.exp.optimizer.beta2), eps=self.args.exp.optimizer.eps)
        else:
            self.optimizer=torch.optim.Adam(list(self.network.parameters()), lr=self.args.exp.lr, betas=(self.args.exp.optimizer.beta1, self.args.exp.optimizer.beta2), eps=self.args.exp.optimizer.eps)

        self.total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print("total_params: ",self.total_params/1e6, "M")

        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)
        
        #resume from checkpoint
        self.latest_checkpoint=None
        resuming=False
        if self.args.exp.finetuning:
            if self.args.exp.base_checkpoint != "None":
                print("Loading base checkpoint:", self.args.exp.base_checkpoint)
                did_it_work =self.resume_from_checkpoint(checkpoint_path=self.args.exp.base_checkpoint)
                if did_it_work:
                    print("Resuming from iteration {}".format(self.it))
                    self.base_checkpoint=self.args.exp.base_checkpoint
                    resuming=True
                else:
                    raise Exception("Could not load base checkpoint")
            else:
                raise Exception("No base checkpoint provided")

        elif self.args.exp.resume:
            if self.args.exp.resume_checkpoint != "None":
                resuming =self.resume_from_checkpoint(checkpoint_path=self.args.exp.resume_checkpoint)
            else:
                resuming =self.resume_from_checkpoint()
            if not resuming:
                print("Could not resume from checkpoint")
                print("training from scratch")
            else:
                print("Resuming from iteration {}".format(self.it))
        else:
            print("No base checkpoint nor resuming provided")

        if not resuming:
            self.it=0
            self.latest_checkpoint=None

        if self.args.logging.log:
            #assert self.args.logging.heavy_log_interval % self.args.logging.save_interval == 0 #sorry for that, I just want to make sure that you are not wasting your time by logging too often, as the tester is only updated with the ema weights from a checkpoint
            self.setup_wandb()
            if self.do_test:
               self.tester.setup_wandb_run(self.wandb_run)
            self.setup_logging_variables()

        if self.args.exp.finetuning:
            self.tester.load_checkpoint(self.base_checkpoint)
            self.tester.it=self.it

        if self.args.exp.LTAS.calculate:
            self.LTAS=self.calculate_LTAS()

        self.save_checkpoint()

    def calculate_LTAS(self):
        print("LTAS computing...")

        nfft=self.args.exp.LTAS.nfft
        win_length=self.args.exp.LTAS.win_length
        hop_length=self.args.exp.LTAS.hop_length

        for i in tqdm(range(self.args.exp.LTAS.num_batches)):
            batch=self.get_batch()
            X=torch.stft(batch, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length).to(self.device), return_complex=True)/torch.sqrt(torch.hann_window(win_length).sum())
    
            L=X.shape[-1]
            Xsum=torch.sum(torch.abs(X)**2, dim=-1).unsqueeze(-1)
            Xsum=torch.mean(Xsum, dim=0)
    
            if i==0:
                Xs = Xsum
                Ls=[L]
            else:
                Xs = torch.cat((Xs, Xsum), dim=-1)
                Ls.append(L)

        Ls=torch.tensor(Ls, dtype=torch.float32).unsqueeze(0).to(self.device)
        X_norm = Xs/Ls
        LTAS = torch.mean(X_norm, dim=-1)
        return LTAS      

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config=omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        config["total_params"]=self.total_params
        self.wandb_run=wandb.init(project=self.args.exp.wandb.project, entity=self.args.exp.wandb.entity, config=config)
        wandb.watch(self.network, log="all", log_freq=self.args.logging.heavy_log_interval) #wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name=os.path.basename(self.args.model_dir)+"_"+self.args.exp.exp_name+"_"+self.wandb_run.id #adding the experiment number to the run name, bery important, I hope this does not crash
    
    def setup_logging_variables(self):

        self.sigma_bins = np.logspace(np.log10(self.args.diff_params.sigma_min), np.log10(self.args.diff_params.sigma_max), num=self.args.logging.num_sigma_bins, base=10)

    def load_state_dict(self, state_dict):
        #print(state_dict)
        return t_utils.load_state_dict(state_dict, network=self.network, ema=self.ema, optimizer=self.optimizer)


    def resume_from_checkpoint(self, checkpoint_path=None, checkpoint_id=None):
        # Resume training from latest checkpoint available in the output director
        if checkpoint_path is not None:
            try:
                checkpoint=torch.load(checkpoint_path, map_location=self.device)
                print(checkpoint.keys())
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=157007 #large number to mean that we loaded somethin, but it is arbitrary
                return self.load_state_dict(checkpoint)
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it=0

            try:
                checkpoint=torch.load(os.path.join(self.args.model_dir,checkpoint_path), map_location=self.device)
                print(checkpoint.keys())
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=157007 #large number to mean that we loaded somethin, but it is arbitrary
                self.network.load_state_dict(checkpoint['ema_model'])
                return True
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it=0
                return False
        else:
            try:
                print("trying to load a project checkpoint")
                print("checkpoint_id", checkpoint_id)
                if checkpoint_id is None:
                    # find latest checkpoint_id
                    save_basename = f"{self.args.exp.exp_name}-*.pt"
                    save_name = f"{self.args.model_dir}/{save_basename}"
                    print(save_name)
                    list_weights = glob(save_name)
                    id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
                    list_ids = [int(id_regex.search(weight_path).groups()[0])
                                for weight_path in list_weights]
                    checkpoint_id = max(list_ids)
                    print(checkpoint_id)
    
                checkpoint = torch.load(
                    f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device)
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=159000 #large number to mean that we loaded somethin, but it is arbitrary
                self.load_state_dict(checkpoint)
                return True
            except Exception as e:
                print(e)
                return False

    def state_dict(self):
        return {
            'it': self.it,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'args': self.args,
        }

    def save_checkpoint(self):
        save_basename = f"{self.args.exp.exp_name}-{self.it}.pt"
        save_name = f"{self.args.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)
        print("saving",save_name)
        if self.args.logging.remove_last_checkpoint:
            try:
                os.remove(self.latest_checkpoint)
                print("removed last checkpoint", self.latest_checkpoint)
            except:
                print("could not remove last checkpoint", self.latest_checkpoint)

        self.latest_checkpoint=save_name

    def process_loss_for_logging(self, error: torch.Tensor, sigma: torch.Tensor):
        """
        This function is used to process the loss for logging. It is used to group the losses by the values of sigma and report them using training_stats.
        args:
            error: the error tensor with shape [batch, audio_len]
            sigma: the sigma tensor with shape [batch]
        """
        #sigma values are ranged between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max. We need to quantize the values of sigma into 10 logarithmically spaced bins between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max
        torch.nan_to_num(error) #not tested might crash
        error=error.detach().cpu().numpy()

        for i in range(len(self.sigma_bins)):
            if i == 0:
                mask = sigma <= self.sigma_bins[i]
            elif i == len(self.sigma_bins)-1:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i-1])
            else:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i-1])
            mask=mask.squeeze(-1).cpu()
            if mask.sum() > 0:
                #find the index of the first element of the mask
                idx = np.where(mask==True)[0][0]
                training_stats.report('error_sigma_'+str(self.sigma_bins[i]),error[idx].mean())

    def get_batch(self):
        # load both clean and noisy data batches
        clean_audio, noisy_audio = next(self.dset)
        clean_audio = clean_audio.to(self.device).to(torch.float32)
        noisy_audio = noisy_audio.to(self.device).to(torch.float32)
        return clean_audio, noisy_audio
    
    def train_step(self):
        # Train step
        it_start_time = time.time()
        #self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad()
        st_time = time.time()
        clean_audio, noisy_audio = self.get_batch()

        # Obtener la salida del modelo y la guía del clasificador
        #model_output = self.network(noisy_audio, self.diff_params.cnoise(sigma))
        #classifier_guidance = clean_audio

        # Calcular la pérdida total con "Classifier Free Guidance"
        #lambda_guidance = self.args.exp.lambda_guidance
        error, sigma = self.diff_params.loss_fn(self.network, clean_audio, noisy_audio)

        if self.args.exp.lossmlp.use:
            loss_u = self.lossmlp(self.diff_params.cnoise(sigma))
            error_norm = error / (loss_u.exp()) + loss_u
            loss = error_norm.mean()
        else:
            loss=error.mean()

        loss.backward()

    if self.it <= self.args.exp.lr_rampup_it:
        for g in self.optimizer.param_groups:
            # learning rate ramp up
            g['lr'] = self.args.exp.lr * min(self.it / max(self.args.exp.lr_rampup_it, 1e-8), 1)

    if self.args.exp.use_grad_clip:
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

        # Update weights.
        self.optimizer.step()

        end_time=time.time()
        if self.args.logging.log:
            self.process_loss_for_logging(error, sigma)

        it_end_time = time.time()
        print("it :", self.it, "time:, ", end_time - st_time, "total_time: ", training_stats.report('it_time', it_end_time - it_start_time), "loss: ", training_stats.report('loss', loss.item()))

    def update_ema(self):
        """Update exponential moving average of self.network weights."""

        ema_rampup = self.args.exp.ema_rampup  #ema_rampup should be set to 10000 in the config file
        ema_rate=self.args.exp.ema_rate #ema_rate should be set to 0.9999 in the config file
        t = self.it * self.args.exp.batch
        with torch.no_grad():
            if t < ema_rampup:
                s = np.clip(t / ema_rampup, 0.0, ema_rate)
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * s + src * (1-s))
            else:
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * ema_rate + src * (1-ema_rate))

    def easy_logging(self):
        """
         Do the simplest logging here. This will be called every 1000 iterations or so
        I will use the training_stats.report function for this, and aim to report the means and stds of the losses in wandb
        """
        training_stats.default_collector.update()
        #Is it a good idea to log the stds of the losses? I think it is not.
        loss_mean=training_stats.default_collector.mean('loss')
        self.wandb_run.log({'loss':loss_mean}, step=self.it)
        loss_std=training_stats.default_collector.std('loss')
        self.wandb_run.log({'loss_std':loss_std}, step=self.it)

        it_time_mean=training_stats.default_collector.mean('it_time')
        self.wandb_run.log({'it_time_mean':it_time_mean}, step=self.it)
        it_time_std=training_stats.default_collector.std('it_time')
        self.wandb_run.log({'it_time_std':it_time_std}, step=self.it)
        
        #here reporting the error respect to sigma. I should make a fancier plot too, with mean and std
        sigma_means=[]
        sigma_stds=[]
        sigma_norm_means=[]
        sigma_norm_stds=[]
        for i in range(len(self.sigma_bins)):
            a=training_stats.default_collector.mean('error_sigma_'+str(self.sigma_bins[i]))
            sigma_means.append(a)
            self.wandb_run.log({'error_sigma_'+str(self.sigma_bins[i]):a}, step=self.it)
            a=training_stats.default_collector.std('error_sigma_'+str(self.sigma_bins[i]))
            sigma_stds.append(a)

            a=training_stats.default_collector.mean('error_norm_sigma_'+str(self.sigma_bins[i]))
            sigma_norm_means.append(a)
            self.wandb_run.log({'error_norm_sigma_'+str(self.sigma_bins[i]):a}, step=self.it)
            a=training_stats.default_collector.std('error_norm_sigma_'+str(self.sigma_bins[i]))
            sigma_norm_stds.append(a)

        
        figure=utils_logging.plot_loss_by_sigma(sigma_means,sigma_stds, self.sigma_bins)
        wandb.log({"loss_dependent_on_sigma": figure}, step=self.it, commit=False)

        figure=utils_logging.plot_loss_by_sigma(sigma_norm_means,sigma_norm_stds, self.sigma_bins)
        wandb.log({"loss_norm_dependent_on_sigma": figure}, step=self.it, commit=True)


        #TODO log here the losses at different noise levels. I don't know if these should be heavy
        #TODO also log here the losses at different frequencies if we are reporting them. same as above

    def heavy_logging(self):
        """
        Do the heavy logging here. This will be called every 10000 iterations or so
        """
        pass
        #if self.do_test:

            #if self.latest_checkpoint is not None:
            #    self.tester.load_checkpoint(self.latest_checkpoint)

            #preds=self.tester.sample_unconditional()


    def training_loop(self):
        
        while True:
            # Accumulate gradients.

            self.train_step()

            self.update_ema()
            
            if self.it>0 and self.it%self.args.logging.save_interval==0 and self.args.logging.save_model:
                #self.save_snapshot() #are the snapshots necessary? I think they are not.
                self.save_checkpoint()

            if self.it>0 and self.it%self.args.logging.heavy_log_interval==0 and self.args.logging.log:
                self.heavy_logging()

            if self.it>0 and self.it%self.args.logging.log_interval==0 and self.args.logging.log:
                self.easy_logging()

            # Update state.
            self.it += 1


    # ----------------------------------------------------------------------------
