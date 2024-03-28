# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
#import PIL.Image
import json
import torch
import random
import glob
import soundfile as sf

#try:
#    import pyspng
#except ImportError:
#    pyspng = None

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.
class AudioSmallFolderDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        overfit=False,
        sigma_data=1.0,
        seed=42 ):
        self.overfit=overfit

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.dset_args=dset_args
        path=dset_args.path
        self.sigma_data=dset_args.sigma_data
        

        filelist=glob.glob(os.path.join(path,"*.wav"))
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"+path

        self.train_samples=filelist
        
       
        self.seg_len=int(seg_len)
        self.fs=fs

        if len(self.train_samples) > 200:
            print("warning: this may be slow")

        lengths=[]
        for f in self.train_samples:
            data, samplerate=sf.read(f)
            length=len(data)
            lengths.append(length/samplerate)
            
        #Normalize lengths to 1
        lengths=np.array(lengths)
        lengths/=np.sum(lengths)
        assert np.sum(lengths)>0.99 and np.sum(lengths)<1.01, "lengths do not sum to 1"
        self.priorities=lengths
        print("priorities",self.priorities)

        if self.overfit:
            file=self.train_samples[0]
            data, samplerate = sf.read(file)
            if len(data.shape)>1 :
                data=np.mean(data,axis=1)
            self.overfit_sample=data[10*samplerate:60*samplerate] #use only 50s


        self.running_mean=0

        if self.sigma_data=="calculate":
            self.calculate_sigma_data()

    def calculate_sigma_data(self):
        N=300 #hardcoded, but that's ok :=>)
        sigmas_data=[]
        for i in range(N):
                #num=random.randint(0,len(self.train_samples)-1)
                #use self.priorities to sample random file
                num=np.random.choice(len(self.train_samples),p=self.priorities)

                #for file in self.train_samples:  
                file=self.train_samples[num]
                data, samplerate = sf.read(file)
                assert(samplerate==self.fs, "wrong sampling rate")
                data_clean=data
                #Stereo to mono
                if len(data.shape)>1 :
                    a=np.random.uniform(0,1)
                    data_clean=data_clean[:,0]*a+data_clean[:,1]*(1-a)
        
                num_frames=np.floor(len(data_clean)/self.seg_len) 
                
                if not self.overfit:
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                else:
                    idx=0
                segment=data_clean[idx:idx+self.seg_len]
                segment=segment.astype('float32')
                #segment/=self.sigma_data
                std=segment.std(-1)
                #check if any in std is 0
                if std<2e-4:
                    raise ValueError("std is 0, skipping")
                sigmas_data.append(std)
       
        sigmas_data=torch.tensor(sigmas_data)
        self.sigma_data=sigmas_data.mean()
        print("sigma_data", self.sigma_data)

    def __iter__(self):
        if self.overfit:
           data_clean=self.overfit_sample
        while True:
            try:
                if not self.overfit:
                    #num=random.randint(0,len(self.train_samples)-1)
                    #use self.priorities to sample random file
                    num=np.random.choice(len(self.train_samples),p=self.priorities)

                    #for file in self.train_samples:  
                    file=self.train_samples[num]
                    data, samplerate = sf.read(file)
                    assert(samplerate==self.fs, "wrong sampling rate")
                    data_clean=data
                    #Stereo to mono
                    if len(data.shape)>1 :
                        a=np.random.uniform(0,1)
                        data_clean=data_clean[:,0]*a+data_clean[:,1]*(1-a)
        
                #normalize
                #no normalization!!
                #data_clean=data_clean/np.max(np.abs(data_clean))
             
                #framify data clean files
                num_frames=np.floor(len(data_clean)/self.seg_len) 
                
                #if num_frames>4:
                #get 8 random batches to be a bit faster
                if not self.overfit:
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                else:
                    idx=0
                segment=data_clean[idx:idx+self.seg_len]
                segment=segment.astype('float32')
                #segment/=self.sigma_data
                std=segment.std(-1)
                #check if any in std is 0
                if std<2e-4:
                    raise ValueError("std is 0, skipping")
                #self.running_mean=0.99*self.running_mean+0.01*std
                #print(self.running_mean)
                segment/=self.sigma_data
                if self.dset_args.apply_random_gain:
                    #apply gain defined in gain_range in dB
                    segment*=10**(np.random.uniform(self.dset_args.gain_range[0],self.dset_args.gain_range[1])/20)
                #b=np.mean(np.abs(segment))
                #segment= (10/(b*np.sqrt(2)))*segment #default rms  of 0.1. Is this scaling correct??
                    
                #let's make this shit a bit robust to input scale
                #scale=np.random.uniform(1.75,2.25)
                #this way I estimage sigma_data (after pre_emph) to be around 1
                
                #segment=10.0**(scale) *segment
                yield  segment
            except Exception as e:
                print(e)
                continue
            #else:
            #    pass


