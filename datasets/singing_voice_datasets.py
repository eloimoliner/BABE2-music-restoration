import os
import numpy as np
import torch
import random
import pandas as pd
import glob
import soundfile as sf

class MultiDataset(torch.utils.data.IterableDataset):
    def __init__(self,
        dset_args,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.path
        path_csv=dset_args.path_csv

        priorities=dset_args.priorities #list
        #normalize the priorities by the sum
        
        num_files=[]

        metafilelist=[]
        for i, p in enumerate(path_csv):
            filelist_df=pd.read_csv(p)
            #convert the dataframe to list
            metafilelist.append(filelist_df.values.tolist())
            num_files.append(len(filelist_df.values.tolist()))

            #filelist=glob.glob(os.path.join(path,"*.wav"))
            assert len(metafilelist[i])>0 , "error in dataloading: empty or nonexistent folder"

        priorities=[priorities[j]*num_files[j] for j in range(len(priorities))]
        priorities=[priorities[j]/sum(priorities) for j in range(len(priorities))]
        priorities=np.array(priorities)
        self.priorities_cumsum=np.cumsum(priorities)
        print(self.priorities_cumsum)

        self.train_samples=metafilelist
       
        self.seg_len=int(dset_args.load_len)
        print(dset_args)
        self.normalize=dset_args.normalize
        self.dset_args=dset_args

    def __iter__(self):
        while True:
            try:
                randnum=np.random.uniform(0,1-1e-8)
                #search for the last element in priorities_cumsum that is smaller than randnum
                #print(self.priorities_cumsum, randnum, np.where(self.priorities_cumsum<randnum))
                try:
                    idx=int(np.where(self.priorities_cumsum<randnum)[0][-1]+1)
                    #print(idx)
                except:
                    idx=0
                    #print("except",idx)

                num=random.randint(0,len(self.train_samples[idx])-1)

                #for file in self.train_samples:  
                file=self.train_samples[idx][num]
                data, samplerate = sf.read(file[0])
                if samplerate>48000:
                    continue
                #assert(samplerate==self.fs, "wrong sampling rate")
                data_clean=data
                #Stereo to mono
                if len(data.shape)>1 :
                    data_clean=np.mean(data_clean,axis=1)

                #print(len(data_clean), self.seg_len)
                if len(data_clean)<self.seg_len:
                    #circular padding with random offset
                    idx=0
                    segment=np.tile(data_clean,(self.seg_len//data_clean.shape[-1]+1))[...,idx:idx+self.seg_len]
                    #random offset (circular shift)
                    segment=np.roll(segment, np.random.randint(0,self.seg_len), axis=-1)
                else:
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                    segment=data_clean[idx:idx+self.seg_len]
                segment=segment.astype('float32')
                if self.normalize:
                    std=segment.std(-1)
                    #check if any in std is 0
                    if std<2e-5:
                        print("std is 0, skipping")
                        continue
                        #raise ValueError("std is 0, skipping")
                    segment/=std

                if self.dset_args.apply_random_gain:
                    #apply gain defined in gain_range in dB
                    segment*=10**(np.random.uniform(self.dset_args.gain_range[0],self.dset_args.gain_range[1])/20)
        
                yield  segment, samplerate

            except Exception as e:
                print(e)
                continue

class AudioFolderDatasetTest_2(torch.utils.data.Dataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        num_samples=4,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.test.path

        filelist=glob.glob(os.path.join(path,"*.wav"))
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"
        self.train_samples=filelist
        self.seg_len=int(seg_len)
        self.fs=fs

        self.test_samples=[]
        self.filenames=[]
        self._fs=[]
        for i in range(num_samples):
            file=self.train_samples[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            data=data.T
            self._fs.append(samplerate)
            if data.shape[-1]>=self.seg_len:
                idx=np.random.randint(0,data.shape[-1]-self.seg_len)
                data=data[...,idx:idx+self.seg_len]
            else:
                idx=0
                data=np.tile(data,(self.seg_len//data.shape[-1]+1))[...,idx:idx+self.seg_len]

            if not dset_args.test.stereo and len(data.shape)>1 :
                data=np.mean(data,axis=1)

            self.test_samples.append(data[...,0:self.seg_len]) #use only 50s


    def __getitem__(self, idx):
        #return self.test_samples[idx]
        return self.test_samples[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)

class AudioFolderDatasetTest(torch.utils.data.Dataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        num_samples=4,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.test.path

        filelist=glob.glob(os.path.join(path,"*.wav"))
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"
        self.train_samples=filelist
        self.seg_len=int(seg_len)
        self.fs=fs

        self.sigma_data=dset_args.sigma_data

        self.test_samples=[]
        self.filenames=[]
        self._fs=[]
        for i in range(num_samples):
            file=self.train_samples[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            data=data.T
            self._fs.append(samplerate)
            print("test",data.shape)
            if data.shape[-1]>=self.seg_len:
                print("cropping")
                idx=np.random.randint(0,data.shape[-1]-self.seg_len)
                segment=data[...,idx:idx+self.seg_len]
            else:
                print("tiling")
                idx=0
                segment=np.tile(data,(self.seg_len//data.shape[-1]+1))[...,idx:idx+self.seg_len]

            if not dset_args.test.stereo and len(data.shape)>1 :
                print("shape",segment.shape)
                segment=np.mean(segment,axis=0)

            std=segment.std(-1)
            print("test",std)
            #check if any in std is 0
            if std<2e-3:
                pass
            else:
                if dset_args.normalize_to_1:
                    segment/=std
                else:
                    segment/=self.sigma_data

                self.test_samples.append(segment) #use only 50s

    def __getitem__(self, idx):
        #return self.test_samples[idx]
        return self.test_samples[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)

class VocalSet_singlesinger(torch.utils.data.IterableDataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=os.path.join(dset_args.path, dset_args.singer_id, "train")
        self.dset_args=dset_args
        self.sigma_data=dset_args.sigma_data

        filelist=glob.glob(os.path.join(path,"*/*/*.wav"))
        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"

        self.train_samples=filelist
       
        self.seg_len=int(seg_len)
        self.fs=fs

        self.running_mean=0


    def __iter__(self):
        while True:
            try:
                num=random.randint(0,len(self.train_samples)-1)
                #for file in self.train_samples:  
                file=self.train_samples[num]
                data, samplerate = sf.read(file)
                #print(file, samplerate)
                assert(samplerate==self.fs, "wrong sampling rate")
                data_clean=data
                #Stereo to mono
                if len(data.shape)>1 :
                    data_clean=np.mean(data_clean,axis=1)
        
                #normalize
                #no normalization!!
                #data_clean=data_clean/np.max(np.abs(data_clean))
             
                #framify data clean files
                num_frames=np.floor(len(data_clean)/self.seg_len) 
                
                #if num_frames>4:
                #get 8 random batches to be a bit faster
                if len(data_clean)<self.seg_len:
                    #circular padding with random offset
                    idx=0
                    segment=np.tile(data_clean,(self.seg_len//data_clean.shape[-1]+1))[...,idx:idx+self.seg_len]
                    #random offset (circular shift)
                    segment=np.roll(segment, np.random.randint(0,self.seg_len), axis=-1)
                else:
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                    segment=data_clean[idx:idx+self.seg_len]
                segment=segment.astype('float32')
                #segment/=self.sigma_data
                std=segment.std(-1)
                #check if any in std is 0
                if std<2e-4:
                    raise ValueError("std is 0, skipping")
                if self.dset_args.normalize_to_1:
                    segment/=std
                else:
                    segment/=self.sigma_data
                #b=np.mean(np.abs(segment))
                #segment= (10/(b*np.sqrt(2)))*segment #default rms  of 0.1. Is this scaling correct??
                if self.dset_args.apply_random_gain:
                        #apply gain defined in gain_range in dB
                        segment*=10**(np.random.uniform(self.dset_args.gain_range[0],self.dset_args.gain_range[1])/20)
                    
                #let's make this shit a bit robust to input scale
                #scale=np.random.uniform(1.75,2.25)
                #this way I estimage sigma_data (after pre_emph) to be around 1
                
                #segment=10.0**(scale) *segment
                #print(segment.std(-1))
                yield  segment
            except Exception as e:
                print(e)
                continue
            #else:
            #    pass
