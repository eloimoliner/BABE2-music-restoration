
import torch
import plotly.express as px
import pandas as pd
import os
import soundfile as sf
import numpy as np
import librosa

def smooth_LTAS(LTAS,f,Noct=1):
    #based on https://github.com/IoSR-Surrey/MatlabToolbox/blob/4bff1bb2da7c95de0ce2713e7c710a0afa70c705/%2Biosr/%2Bdsp/smoothSpectrum.m


    def gauss_f(f_x,F,Noct):
        sigma = (F/Noct)/np.pi
        g = torch.exp(-(((f_x-F)**2)/(2*(sigma**2))))
        g = g/torch.sum(g)
        return g


    x_oct = LTAS.clone()
    if Noct>0:
        for i in range(1,len(f)):
            g = gauss_f(f,f[i],Noct)
            g=g.to(LTAS.device)
            x_oct[i] = torch.sum(g*LTAS)
        if torch.all(LTAS>=0):
            x_oct[x_oct<0] = 0

    return x_oct


def compute_LTAS(path=None, x=None,audio_files=None,sample_rate=22050,nfft=2048, hop_length=512, win_length=2048, normalize=None, sqrt=False):
    #sample_rate = 22050
    #nfft = 2048
    #win_length = 2048
    #hop_length = 512

    if audio_files is None:
        length=1
        assert x is not None, "x should be provided if audio_files is not provided"
    else:
        length=len(audio_files)

    for i in range(length):
        #read audio file
        if x is None:
            if path is None:
                #if audio_files is a pandas dataframe
                if isinstance(audio_files, pd.DataFrame):
                    x, sr = sf.read(os.path.join(audio_files.iloc[i,0]))
                else:
                    x.sr = sf.read(os.path.join(path,audio_files[i]))
            else:
                if isinstance(audio_files, pd.DataFrame):
                    x, sr = sf.read(os.path.join(path,audio_files.iloc[i,0]))
                else:
                    x, sr = sf.read(os.path.join(path,audio_files[i]))
        else:
            x=x
            sr=sample_rate #assuming x is already at sample_rate
    
        if len(x.shape)==2:
            x=np.mean(x, axis=1)
    
        if sr!=sample_rate:
            x=librosa.resample(x, orig_sr=sr, target_sr=sample_rate)

        x=torch.tensor(x, dtype=torch.float32)

        if normalize is not None:
            std=x.std()
            x=normalize*x/x.std()
        else:
            std=1
    
        X=torch.stft(x, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length), return_complex=True)/torch.sqrt(torch.hann_window(win_length).sum())
    
        L=X.shape[-1]
        if sqrt:
            Xsum=torch.sqrt(torch.sum(torch.abs(X)**2, dim=1).unsqueeze(-1))
        else:
            Xsum=torch.sum(torch.abs(X)**2, dim=1).unsqueeze(-1)
    
        if i==0:
            Xs = Xsum
            Ls=[L]
        else:
            Xs = torch.cat((Xs, Xsum), dim=1)
            Ls.append(L)
    
    Ls=torch.tensor(Ls, dtype=torch.float32).unsqueeze(0)
    X_norm = Xs/Ls
    X_mean = torch.mean(X_norm, dim=-1)
    return X_mean, std

def apply_filter(x, H, NFFT):
    '''
    '''
    X=apply_stft(x, NFFT)
    xrec=apply_filter_istft(X, H, NFFT)
    xrec=xrec[:,:x.shape[-1]]

    return xrec

def apply_stft(x, NFFT):
    '''
    '''
    #hamming window
    window = torch.hamming_window(window_length=NFFT)
    window=window.to(x.device)

    x=torch.cat((x, torch.zeros(*x.shape[:-1],NFFT).to(x.device)),1) #is padding necessary?
    X = torch.stft(x, NFFT, hop_length=NFFT//2,  window=window,  center=False, onesided=True, return_complex=True)
    X=torch.view_as_real(X)

    return X

def apply_filter_istft(X, H, NFFT):
    '''
    '''
    #hamming window
    window = torch.hamming_window(window_length=NFFT)
    window=window.to(X.device)

    X=X*H.unsqueeze(-1).unsqueeze(-1).expand(X.shape)
    X=torch.view_as_complex(X)
    x=torch.istft(X, NFFT, hop_length=NFFT//2,  window=window, center=False, return_complex=False)

    return x

def design_filter_G(fc, A, G, f):
    """
    fc: cutoff frequency 
        if fc is a scalar, the filter has one slopw
        if fc is a list of scalars, the filter has multiple slopes
    A: attenuation in dB
        if A is a scalar, the filter has one slopw
        if A is a list of scalars, the filter has multiple slopes
    """
    multiple_slopes=False
    #check if fc and A are lists
    if isinstance(fc, list) and isinstance(A, list):
        multiple_slopes=True
    #check if fc is a tensor and A is a tensor
    try:
        if fc.shape[0]>1:
            multiple_slopes=True
    except:
        pass

    if multiple_slopes:
        H=torch.zeros(f.shape).to(f.device)
        H[f<fc[0]]=1
        H[f>=fc[0]]=10**(A[0]*torch.log2(f[f>=fc[0]]/fc[0])/20)
        for i in range(1,len(fc)):
            #find the index of the first frequency that is greater than fc[i]
            #fix=torch.where(f>=fc[i])[0][0]
            #print(fc[i],fix)
            #H[f>=fc[i]]=H[fix]-10**(A[i]*torch.log2(f[f>=fc[i]]/fc[i])/20)
            H[f>=fc[i]]=10**(A[i]*torch.log2(f[f>=fc[i]]/fc[i])/20)*H[f>=fc[i]][0]
        #apply the gain (G is in dB)
        H=H*10**(G/20)
        #return H
    else:
        #if fc and A are scalars
        H=torch.zeros(f.shape).to(f.device)
        H[f<fc]=1
        H[f>=fc]=10**(A*torch.log2(f[f>=fc]/fc)/20)
        H=H*10**(G/20)
    return H

def design_filter_2(fref, params, f, block_low_freq=False):
    """
    fc: cutoff frequency 
        if fc is a scalar, the filter has one slopw
        if fc is a list of scalars, the filter has multiple slopes
    A: attenuation in dB
        if A is a scalar, the filter has one slopw
        if A is a list of scalars, the filter has multiple slopes
    """
    #fref=params[0]
    fc_p=params[0]
    fc_m=params[1]
    A_p=params[2]
    A_m=params[3]

    assert (fc_p<=fref).any()==False, f"fc_p must be greater than fref: {fc_p}, {fref}"
    assert (fc_m>=fref).any()==False, f"fc_m must be smaller than fre: {fc_m}, {fref}"

    assert (fc_m <= f[1]).any()==False, f"fc_m must be greater than the minimum frequency: {fc_m}, {f[1]}"
    assert (fc_p >= f[-1]).any()==False, f"fc_p must be smaller than the maximum frequency: {fc_p}, {f[-1]}"

    f=f[1:]
    H=torch.ones(f.shape).to(f.device)

    #H[f<fref]=1


    H[f>=fref]=10**(A_p[0]*torch.log2(f[f>=fref]/fref)/20)
    for i in range(0,len(fc_p)):
        #find the index of the first frequency that is greater than fc[i]
        H[f>=fc_p[i]]=10**(A_p[i+1]*torch.log2(f[f>=fc_p[i]]/fc_p[i])/20)*H[f>=fc_p[i]][0]

    if block_low_freq:
        pass
    else:
        H[f<fref]=10**(A_m[0]*torch.log2(f[f<fref]/fref)/20)*H[f<fref][-1]
        for i in range(0,len(fc_m)):
            #find the index of the first frequency that is greater than fc[i]
            H[f<fc_m[i]]=10**(A_m[i+1]*torch.log2(f[f<fc_m[i]]/fc_m[i])/20)*H[f<fc_m[i]][-1]
    
    #concatenate the DC component
    H=torch.cat((torch.zeros(1).to(f.device), H),0)

    return H

def design_filter_3( params, f, block_low_freq=False):
    """
    fc: cutoff frequency 
        if fc is a scalar, the filter has one slopw
        if fc is a list of scalars, the filter has multiple slopes
    A: attenuation in dB
        if A is a scalar, the filter has one slopw
        if A is a list of scalars, the filter has multiple slopes
    """
    fref=params[0]
    fc_p=params[1]
    fc_m=params[2]
    A_p=params[3]
    A_m=params[4]

    assert (fc_p<=fref).any()==False, f"fc_p must be greater than fref: {fc_p}, {fref}"
    assert (fc_m>=fref).any()==False, f"fc_m must be smaller than fre: {fc_m}, {fref}"

    assert (fc_m <= f[1]).any()==False, f"fc_m must be greater than the minimum frequency: {fc_m}, {f[1]}"
    assert (fc_p >= f[-1]).any()==False, f"fc_p must be smaller than the maximum frequency: {fc_p}, {f[-1]}"

    f=f[1:]
    H=torch.ones(f.shape).to(f.device)

    #H[f<fref]=1


    H[f>=fref]=10**(A_p[0]*torch.log2(f[f>=fref]/fref)/20)
    for i in range(0,len(fc_p)):
        #find the index of the first frequency that is greater than fc[i]
        H[f>=fc_p[i]]=10**(A_p[i+1]*torch.log2(f[f>=fc_p[i]]/fc_p[i])/20)*H[f>=fc_p[i]][0]

    if block_low_freq:
        pass
    else:
        H[f<fref]=10**(A_m[0]*torch.log2(f[f<fref]/fref)/20)*H[f<fref][-1]
        for i in range(0,len(fc_m)):
            #find the index of the first frequency that is greater than fc[i]
            H[f<fc_m[i]]=10**(A_m[i+1]*torch.log2(f[f<fc_m[i]]/fc_m[i])/20)*H[f<fc_m[i]][-1]
    
    #concatenate the DC component
    H=torch.cat((torch.zeros(1).to(f.device), H),0)

    return H

def design_filter_BABE(params, f):
    """
    fc: cutoff frequency 
        if fc is a scalar, the filter has one slopw
        if fc is a list of scalars, the filter has multiple slopes
    A: attenuation in dB
        if A is a scalar, the filter has one slopw
        if A is a list of scalars, the filter has multiple slopes
    """
    fc=params[0]
    A=params[1]
    multiple_slopes=False
    #check if fc and A are lists
    if isinstance(fc, list) and isinstance(A, list):
        multiple_slopes=True
    #check if fc is a tensor and A is a tensor
    try:
        if fc.shape[0]>1:
            multiple_slopes=True
    except:
        pass

    if multiple_slopes:
        H=torch.zeros(f.shape).to(f.device)
        H[f<fc[0]]=1
        H[f>=fc[0]]=10**(A[0]*torch.log2(f[f>=fc[0]]/fc[0])/20)
        for i in range(1,len(fc)):
            #find the index of the first frequency that is greater than fc[i]
            #fix=torch.where(f>=fc[i])[0][0]
            #print(fc[i],fix)
            #H[f>=fc[i]]=H[fix]-10**(A[i]*torch.log2(f[f>=fc[i]]/fc[i])/20)
            H[f>=fc[i]]=10**(A[i]*torch.log2(f[f>=fc[i]]/fc[i])/20)*H[f>=fc[i]][0]

        #return H
    else:
        #if fc and A are scalars
        H=torch.zeros(f.shape).to(f.device)
        H[f<fc]=1
        H[f>=fc]=10**(A*torch.log2(f[f>=fc]/fc)/20)
    return H

#def design_filter(fc, A, f):
#    H=torch.zeros(f.shape).to(f.device)
#    H[f<fc]=1
#    H[f>=fc]=10**(A*torch.log2(f[f>=fc]/fc)/20)
#    return H




def apply_filter_and_norm_STFTmag(X,Xref, H):
    #X: (N,513, T) "clean" example
    #Xref: (N,513, T)  observations
    #H: (513,) filter

    #get the absolute value of the STFT
    X=torch.sqrt(X[...,0]**2+X[...,1]**2)
    Xref=torch.sqrt(Xref[...,0]**2+Xref[...,1]**2)

    X=X*H.unsqueeze(-1).expand(X.shape)
    norm=torch.linalg.norm(X.reshape(-1)-Xref.reshape(-1),ord=2)
    return norm

def apply_norm_filter(H,H2):
    norm=torch.linalg.norm(H.reshape(-1)-H2.reshape(-1),ord=2)

    return norm

def apply_norm_STFT_fweighted(y,den_rec, freq_weight="linear", NFFT=1024):
    #X: (N,513, T) "clean" example
    #Xref: (N,513, T)  observations
    #H: (513,) filter
    X=apply_stft(den_rec, NFFT)
    Xref=apply_stft(y, NFFT)

    #get the absolute value of the STFT
    #X=torch.sqrt(X[...,0]**2+X[...,1]**2)
    #Xref=torch.sqrt(Xref[...,0]**2+Xref[...,1]**2)

    freqs=torch.linspace(0, 1, X.shape[1]).to(X.device).unsqueeze(-1)
    #print(X.shape, Xref.shape, freqs.shape)
        
    #apply frequency weighting to the cost function
    if freq_weight=="linear":
        X=X*freqs.unsqueeze(-1).expand(X.shape)
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)
    elif freq_weight=="None":
        pass
    elif freq_weight=="log":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="sqrt":
        X=X*torch.sqrt(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.sqrt(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="log2":
        X=X*torch.log2(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log2(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="log10":
        X=X*torch.log10(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log10(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="cubic":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**3
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**3
    elif freq_weight=="quadratic":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**2
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**2
    elif freq_weight=="logcubic":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape)**3)
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape)**3)
    elif freq_weight=="logquadratic":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape)**2)
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape)**2)
    elif freq_weight=="squared":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**4
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**4

    norm=torch.linalg.norm(X.reshape(-1)-Xref.reshape(-1),ord=2)
    return norm
def apply_norm_STFTmag_fweighted(y,den_rec, freq_weight="linear", NFFT=1024, logmag=False):
    #X: (N,513, T) "clean" example
    #Xref: (N,513, T)  observations
    #H: (513,) filter
    X=apply_stft(den_rec, NFFT)
    Xref=apply_stft(y, NFFT)

    #get the absolute value of the STFT
    X=torch.sqrt(X[...,0]**2+X[...,1]**2)
    Xref=torch.sqrt(Xref[...,0]**2+Xref[...,1]**2)

    freqs=torch.linspace(0, 1, X.shape[1]).to(X.device)
    #apply frequency weighting to the cost function
    if freq_weight=="linear":
        X=X*freqs.unsqueeze(-1).expand(X.shape)
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)
    elif freq_weight=="None":
        pass
    elif freq_weight=="log":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="sqrt":
        X=X*torch.sqrt(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.sqrt(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="log2":
        X=X*torch.log2(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log2(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="log10":
        X=X*torch.log10(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log10(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="cubic":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**3
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**3
    elif freq_weight=="quadratic":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**2
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**2
    elif freq_weight=="logcubic":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape)**3)
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape)**3)
    elif freq_weight=="logquadratic":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape)**2)
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape)**2)
    elif freq_weight=="squared":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**4
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**4

    if logmag==True:
        norm=torch.linalg.norm(torch.log10(X.reshape(-1)+1e-8)-torch.log10(Xref.reshape(-1)+1e-8),ord=2)
    else:
        norm=torch.linalg.norm(X.reshape(-1)-Xref.reshape(-1),ord=2)
    return norm

def apply_filter_and_norm_STFTmag_fweighted(X,Xref, H, freq_weight="linear"):
    #X: (N,513, T) "clean" example
    #Xref: (N,513, T)  observations
    #H: (513,) filter

    #get the absolute value of the STFT
    X=torch.sqrt(X[...,0]**2+X[...,1]**2)
    Xref=torch.sqrt(Xref[...,0]**2+Xref[...,1]**2)

    X=X*H.unsqueeze(-1).expand(X.shape)
    freqs=torch.linspace(0, 1, X.shape[1]).to(X.device)
    #apply frequency weighting to the cost function
    if freq_weight=="linear":
        X=X*freqs.unsqueeze(-1).expand(X.shape)
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)
    elif freq_weight=="None":
        pass
    elif freq_weight=="log":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="sqrt":
        X=X*torch.sqrt(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.sqrt(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="log2":
        X=X*torch.log2(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log2(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="log10":
        X=X*torch.log10(freqs.unsqueeze(-1).expand(X.shape))
        Xref=Xref*torch.log10(freqs.unsqueeze(-1).expand(Xref.shape))
    elif freq_weight=="cubic":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**3
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**3
    elif freq_weight=="quadratic":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**2
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**2
    elif freq_weight=="logcubic":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape)**3)
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape)**3)
    elif freq_weight=="logquadratic":
        X=X*torch.log2(1+freqs.unsqueeze(-1).expand(X.shape)**2)
        Xref=Xref*torch.log2(1+freqs.unsqueeze(-1).expand(Xref.shape)**2)
    elif freq_weight=="squared":
        X=X*freqs.unsqueeze(-1).expand(X.shape)**4
        Xref=Xref*freqs.unsqueeze(-1).expand(Xref.shape)**4

    norm=torch.linalg.norm(X.reshape(-1)-Xref.reshape(-1),ord=2)
    return norm


def plot_single_filter_BABE2_3(est_filter, freqs):
    f=freqs
    H=design_filter_3( est_filter, f)
    fig=px.line(x=f.cpu(),y=20*torch.log10(H.cpu().detach()), log_x=True , title='Frequency response of a low pass filter', labels={'x':'Frequency (Hz)', 'y':'Magnitude (dB)'})
    #plot the reference frequency response
    #define range
    fig.update_yaxes(range=[-100, 30])
    return fig
