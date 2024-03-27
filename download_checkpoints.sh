#!/bin/bash

#script to download the trained checkpoitntrained checkpoints

cd experiments

#MAESTRO (piano)
wget http://research.spa.aalto.fi/publications/papers/dafx-babe2/checkpoints/MAESTRO_22kHz_8s-850kits.pt


#pretrained on singing voice datasets 
#wget http://research.spa.aalto.fi/publications/papers/dafx-babe2/checkpoints/singing_voice_pretrain_44kHz_6s-325kits.pt

##VocalSet

#male 2 (used for Enrico Caruso)
wget http://research.spa.aalto.fi/publications/papers/dafx-babe2/checkpoints/VocalSet_male2_44kHz_6s-8kits.pt

#male 11 (used for Beniamo Gigli)
#wget http://research.spa.aalto.fi/publications/papers/dafx-babe2/checkpoints/VocalSet_male11_44kHz_6s-8kits.pt

#Female 1 (usef for Nellie Melba)
#wget http://research.spa.aalto.fi/publications/papers/dafx-babe2/checkpoints/VocalSet_female1_44kHz_6s-8kits.pt

#Female 5 (usef for Adelina Patti)
#wget http://research.spa.aalto.fi/publications/papers/dafx-babe2/checkpoints/VocalSet_female5_44kHz_6s-8kits.pt
