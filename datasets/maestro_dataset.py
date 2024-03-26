import os
import numpy as np
import torch
import random
import pandas as pd
import soundfile as sf

class MaestroDataset_fs(torch.utils.data.IterableDataset):
    def __init__(self,
        dset_args,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.path
        years=dset_args.years

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]=="train"]
        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x)     , na_action='ignore')


        self.train_samples=filelist.to_list()
       
        self.seg_len=int(dset_args.load_len)

    def __iter__(self):
        while True:
            num=random.randint(0,len(self.train_samples)-1)
            #for file in self.train_samples:  
            file=self.train_samples[num]
            data, samplerate = sf.read(file)
            #print(file,samplerate)

            data_clean=data
            #Stereo to mono
            if len(data.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
    
            #normalize
            #no normalization!!
            #data_clean=data_clean/np.max(np.abs(data_clean))
         
            #framify data clean files

            num_frames=np.floor(len(data_clean)/self.seg_len) 
            
            if num_frames>4:
                for i in range(8):
                    #get 8 random batches to be a bit faster
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                    segment=data_clean[idx:idx+self.seg_len]
                    segment=segment.astype('float32')
                    #b=np.mean(np.abs(segment))
                    #segment= (10/(b*np.sqrt(2)))*segment #default rms  of 0.1. Is this scaling correct??
                     
                    #let's make this shit a bit robust to input scale
                    #scale=np.random.uniform(1.75,2.25)
                    #this way I estimage sigma_data (after pre_emph) to be around 1
                   
                    #segment=10.0**(scale) *segment
                    yield  segment, samplerate
            else:
                pass

class MaestroDatasetTestChunks(torch.utils.data.Dataset):
    def __init__(self,
        dset_args,
        num_samples=4,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.path
        years=dset_args.years

        self.seg_len=int(dset_args.load_len)

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]=="test"]
        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x)     , na_action='ignore')


        self.filelist=filelist.to_list()

        self.test_samples=[]
        self.filenames=[]
        self.f_s=[]
        for i in range(num_samples):
            file=self.filelist[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            if len(data.shape)>1 :
                data=np.mean(data,axis=1)

            self.test_samples.append(data[10*samplerate:10*samplerate+self.seg_len]) #use only 50s
            self.f_s.append(samplerate)
       

    def __getitem__(self, idx):
        return self.test_samples[idx], self.f_s[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)

