# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import json
import torch
import random
import glob
import soundfile as sf

class AudioPairFolderDataset(torch.utils.data.IterableDataset):
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

        clean_filelist=glob.glob(os.path.join(path,"clean","*.wav"))
        noisy_filelist=glob.glob(os.path.join(path,"noisy","*.wav"))
        assert len(clean_filelist)>0 and len(noisy_filelist)>0, "error in dataloading: empty or nonexistent folder"+path

        self.train_samples=list(zip(clean_filelist, noisy_filelist))
        
        self.seg_len=int(seg_len)
        self.fs=fs

        if len(self.train_samples) > 200:
            print("warning: this may be slow")

        lengths=[]
        for f in self.train_samples:
            data, samplerate=sf.read(f[0])  # assuming clean and noisy files have same length
            length=len(data)
            lengths.append(length/samplerate)
            
        lengths=np.array(lengths)
        lengths/=np.sum(lengths)
        assert np.sum(lengths)>0.99 and np.sum(lengths)<1.01, "lengths do not sum to 1"
        self.priorities=lengths
        print("priorities",self.priorities)

        if self.overfit:
            file=self.train_samples[0]
            clean_data, samplerate = sf.read(file[0])
            noisy_data, _ = sf.read(file[1])
            if len(clean_data.shape)>1 :
                clean_data=np.mean(clean_data,axis=1)
                noisy_data=np.mean(noisy_data,axis=1)
            self.overfit_sample=(clean_data[10*samplerate:60*samplerate], noisy_data[10*samplerate:60*samplerate]) #use only 50s

        self.running_mean=0

        if self.sigma_data=="calculate":
            self.calculate_sigma_data()

    def calculate_sigma_data(self):
        N=300
        sigmas_data=[]
        for i in range(N):
                num=np.random.choice(len(self.train_samples),p=self.priorities)
                file=self.train_samples[num]
                clean_data, samplerate = sf.read(file[0])
                noisy_data, _ = sf.read(file[1])
                assert(samplerate==self.fs, "wrong sampling rate")
                data_clean=clean_data
                if len(clean_data.shape)>1 :
                    a=np.random.uniform(0,1)
                    data_clean=data_clean[:,0]*a+data_clean[:,1]*(1-a)
        
                num_frames=np.floor(len(data_clean)/self.seg_len) 
                
                if not self.overfit:
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                else:
                    idx=0
                segment=data_clean[idx:idx+self.seg_len]
                segment=segment.astype('float32')
                std=segment.std(-1)
                if std<2e-4:
                    raise ValueError("std is 0, skipping")
                sigmas_data.append(std)
       
        sigmas_data=torch.tensor(sigmas_data)
        self.sigma_data=sigmas_data.mean()
        print("sigma_data", self.sigma_data)

    def __iter__(self):
        if self.overfit:
           data_clean=self.overfit_sample[0]
           data_noisy=self.overfit_sample[1]
        while True:
            try:
                if not self.overfit:
                    num=np.random.choice(len(self.train_samples),p=self.priorities)
                    file=self.train_samples[num]
                    clean_data, samplerate = sf.read(file[0])
                    noisy_data, _ = sf.read(file[1])
                    assert(samplerate==self.fs, "wrong sampling rate")
                    data_clean=clean_data
                    data_noisy=noisy_data
                    if len(clean_data.shape)>1 :
                        a=np.random.uniform(0,1)
                        data_clean=data_clean[:,0]*a+data_clean[:,1]*(1-a)
                        data_noisy=data_noisy[:,0]*a+data_noisy[:,1]*(1-a)
        
                num_frames=np.floor(len(data_clean)/self.seg_len) 
                
                if not self.overfit:
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                else:
                    idx=0
                clean_segment=data_clean[idx:idx+self.seg_len]
                noisy_segment=data_noisy[idx:idx+self.seg_len]
                clean_segment=clean_segment.astype('float32')
                noisy_segment=noisy_segment.astype('float32')
                std=clean_segment.std(-1)
                if std<2e-4:
                    raise ValueError("std is 0, skipping")
                clean_segment/=self.sigma_data
                noisy_segment/=self.sigma_data
                if self.dset_args.apply_random_gain:
                    gain=10**(np.random.uniform(self.dset_args.gain_range[0],self.dset_args.gain_range[1])/20)
                    clean_segment*=gain
                    noisy_segment*=gain
                yield  (clean_segment, noisy_segment)
            except Exception as e:
                print(e)
                continue
