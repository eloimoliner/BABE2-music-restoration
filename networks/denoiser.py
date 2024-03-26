import torch.nn as nn
import math as m
import torch
#import torchaudio
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

#def build_model_denoise(unet_args=None):
#
#    inputs=Input(shape=(None, None,2))
#
#    outputs_stage_2,outputs_stage_1=MultiStage_denoise(unet_args=unet_args)(inputs)
#
#    #Encapsulating MultiStage_denoise in a keras.Model object
#    model= tf.keras.Model(inputs=inputs,outputs=[outputs_stage_2, outputs_stage_1])

#    return model

class DenseBlock(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, N] 
    DenseNet Block consisting of "num_layers" densely connected convolutional layers
    '''
    def __init__(self, num_layers,N0, N, ksize):
        '''
        num_layers:     number of densely connected conv. layers
        N:              Number of filters (same in each layer) 
        ksize:          Kernel size (same in each layer) 
        '''
        super(DenseBlock, self).__init__()

        self.H=nn.ModuleList()
        self.num_layers=num_layers

        for i in range(num_layers):
            if i==0:   
                Nin=N0
            else:
                Nin=N0+i*N
             
            self.H.append(nn.Sequential(
                                nn.Conv2d(Nin,N,
                                      kernel_size=ksize,
                                      stride=1,
                                      padding='same',
                                      padding_mode='reflect',
                                      ),
                                nn.ELU()        ))

    def forward(self, x):
        x_ = self.H[0](x)
        if self.num_layers>1:
            for h in self.H[1:]:
                x = torch.cat((x_, x), 1)
                #x_=tf.pad(x, self.padding_modes_1, mode='SYMMETRIC')
                x_ = h(x)  
                #add elu here

        return x_


class FinalBlock(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, 2] 
    Final block. Basiforwardy, a 3x3 conv. layer to map the output features to the output complex spectrogram.

    '''
    def __init__(self, N0):
        super(FinalBlock, self).__init__()
        ksize=(3,3)
        self.conv2=nn.Conv2d(N0,out_channels=2,
                      kernel_size=ksize,
                      stride=1, 
                      padding='same',
                      padding_mode='reflect')


    def forward(self, inputs ):

        pred=self.conv2(inputs)

        return pred

class SAM(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, N] , [B, T, F, N]
    Supervised Attention Module:
    The purpose of SAM is to make the network only propagate the most relevant features to the second stage, discarding the less useful ones.
    The estimated residual noise signal is generated from the U-Net output features by means of a 3x3 convolutional layer. 
    The first stage output is then calculated adding the original input spectrogram to the residual noise. 
    The attention-guided features are computed using the attention masks M, which are directly calculated from the first stage output with a 1x1 convolution and a sigmoid function. 

    '''
    def __init__(self, n_feat):
        super(SAM, self).__init__()

        ksize=(3,3)
        self.conv1 = nn.Conv2d(n_feat,out_channels=n_feat,
                      kernel_size=ksize,
                      stride=1, 
                      padding='same',
                      padding_mode='reflect')
        ksize=(3,3)
        self.conv2=nn.Conv2d( n_feat,2,
                      kernel_size=ksize,
                      stride=1, 
                      padding='same',
                      padding_mode='reflect')

        ksize=(3,3)
        self.conv3 = nn.Conv2d(2,n_feat,
                      kernel_size=ksize,
                      stride=1, 
                      padding='same',
                      padding_mode='reflect')

        #self.cropadd=CropAddBlock()

    def forward(self, inputs, input_spectrogram):
        x1 = self.conv1(inputs)
        x=self.conv2(inputs)

        #residual prediction
        pred = torch.add(x, input_spectrogram) #features to next stage

        M=self.conv3(pred)

        M= torch.sigmoid(M)
        x1=torch.multiply(x1, M)
        x1 = torch.add(x1, inputs) #features to next stage

        return x1, pred


class AddFreqEncoding(nn.Module):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim):
        super(AddFreqEncoding, self).__init__()
        pi=torch.pi
        self.f_dim=f_dim #f_dim is fixed
        n=torch.arange(start=0,end=f_dim)/(f_dim-1)
        # n=n.type(torch.FloatTensor)
        coss=torch.cos(pi*n)
        f_channel = torch.unsqueeze(coss, -1) #(1025,1)
        self.fembeddings= f_channel
        
        for k in range(1,10):   
            coss=torch.cos(2**k*pi*n)
            f_channel = torch.unsqueeze(coss, -1) #(1025,1)
            self.fembeddings=torch.cat((self.fembeddings,f_channel),-1) #(1025,10)

        self.fembeddings=nn.Parameter(self.fembeddings)
        #self.register_buffer('fembeddings_const', self.fembeddings)

    

    def forward(self, input_tensor):

        batch_size_tensor = input_tensor.shape[0]  # get batch size
        time_dim = input_tensor.shape[2]  # get time dimension

        fembeddings_2 = torch.broadcast_to(self.fembeddings, [batch_size_tensor, time_dim, self.f_dim, 10])
        fembeddings_2=fembeddings_2.permute(0,3,1,2)
    
        
        return torch.cat((input_tensor,fembeddings_2),1)  #(batch,12,427,1025)


class Decoder(nn.Module):
    '''
    [B, T, F, N] , skip connections => [B, T, F, N]  
    Decoder side of the U-Net subnetwork.
    '''
    def __init__(self, Ns, Ss, unet_args):
        super(Decoder, self).__init__()

        self.Ns=Ns
        self.Ss=Ss
        self.depth=unet_args.depth

        self.dblocks=nn.ModuleList()
        for i in range(self.depth):
            self.dblocks.append(D_Block(layer_idx=i,N0=self.Ns[i+1] ,N=self.Ns[i], S=self.Ss[i],num_tfc=unet_args.num_tfc))

    def forward(self,inputs, contracting_layers):
        x=inputs
        for i in range(self.depth,0,-1):
            x=self.dblocks[i-1](x, contracting_layers[i-1])
        return x 

class Encoder(nn.Module):

    '''
    [B, T, F, N] => skip connections , [B, T, F, N_4]  
    Encoder side of the U-Net subnetwork.
    '''
    def __init__(self,N0, Ns, Ss, unet_args):
        super(Encoder, self).__init__()
        self.Ns=Ns
        self.Ss=Ss
        self.depth=unet_args.depth

        self.contracting_layers = {}

        self.eblocks=nn.ModuleList()
        for i in range(self.depth):
            if i==0:
                Nin=N0
            else:
                Nin=self.Ns[i]

            self.eblocks.append(E_Block(layer_idx=i,N0=Nin,N01=self.Ns[i],N=self.Ns[i+1],S=self.Ss[i], num_tfc=unet_args.num_tfc))

        self.i_block=I_Block(self.Ns[self.depth],self.Ns[self.depth],unet_args.num_tfc)

    def forward(self, inputs):
        x=inputs
        for i in range(self.depth):

            x, x_contract=self.eblocks[i](x)
        
            self.contracting_layers[i] = x_contract #if remove 0, correct this


        x=self.i_block(x)

        return x, self.contracting_layers

class MultiStage_denoise(nn.Module):

    def __init__(self,  unet_args=None):
        super(MultiStage_denoise, self).__init__()
        self.depth=unet_args.depth
        Nin=2
        if unet_args.use_fencoding:
            self.freq_encoding=AddFreqEncoding(unet_args.f_dim)
            Nin=12 #hardcoded
        self.use_sam=unet_args.use_SAM
        self.use_fencoding=unet_args.use_fencoding
        self.num_stages=unet_args.num_stages
        #Encoder
        self.Ns= [64,64,64,128,128,256,512] 
        self.Ss= [(2,2),(2,2),(2,2),(2,2),(2,2),(2,2)]
        
        #initial feature extractor
        ksize=(7,7)

        self.conv2d_1 = nn.Sequential(nn.Conv2d(Nin,self.Ns[0],
                      kernel_size=ksize,
                      padding='same',
                      padding_mode='reflect'),
                      nn.ELU())
                        
        self.encoder_s1=Encoder(self.Ns[0],self.Ns, self.Ss, unet_args)
        self.decoder_s1=Decoder(self.Ns, self.Ss, unet_args)

        self.cropconcat = CropConcatBlock()
        #self.cropadd = CropAddBlock()

        self.finalblock=FinalBlock(self.Ns[0])

        if self.num_stages>1:
            self.sam_1=SAM(self.Ns[0])

            #initial feature extractor
            ksize=(7,7)

            self.conv2d_2 =nn.Sequential(
                                 nn.Conv2d(Nin,self.Ns[0],
                                 kernel_size=ksize,
                                 stride=1, 
                                 padding='same',
                                 padding_mode='reflect'),
                                 nn.ELU())
            

            self.encoder_s2=Encoder(2*self.Ns[0],self.Ns, self.Ss, unet_args)
            self.decoder_s2=Decoder(self.Ns, self.Ss, unet_args)

    def forward(self, inputs):
        if self.use_fencoding:
            x_w_freq=self.freq_encoding(inputs)   #None, None, 1025, 12 
        else:
            x_w_freq=inputs

        #intitial feature extractor
        x=self.conv2d_1(x_w_freq) #None, None, 1025, 32

        x, contracting_layers_s1= self.encoder_s1(x)
        #decoder

        feats_s1 =self.decoder_s1(x, contracting_layers_s1) #None, None, 1025, 32 features

        if self.num_stages>1:        
            #SAM module
            Fout, pred_stage_1=self.sam_1(feats_s1,inputs)
                
            #intitial feature extractor
            x=self.conv2d_2(x_w_freq)
    
            if self.use_sam:
                x = torch.cat((x, Fout), 1)
            else:
                x = torch.cat((x,feats_s1), 1)


            x, contracting_layers_s2= self.encoder_s2(x)


            feats_s2=self.decoder_s2(x, contracting_layers_s2) #None, None, 1025, 32 features
            
            #consider implementing a third stage?

            pred_stage_2=self.finalblock(feats_s2) 
            return pred_stage_2, pred_stage_1
        else:             
            pred_stage_1=self.finalblock(feats_s1) 
            return pred_stage_1
            
class I_Block(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, N] 
    Intermediate block:
    Basiforwardy, a densenet block with a residual connection
    '''
    def __init__(self,N0,N, num_tfc, **kwargs):
        super(I_Block, self).__init__(**kwargs)

        ksize=(3,3)
        self.tfc=DenseBlock(num_tfc,N0,N,ksize)

        self.conv2d_res= nn.Conv2d(N0,N,
                                      kernel_size=(1,1),
                                      stride=1,
                                      padding='same',
                                      padding_mode='reflect')

    def forward(self,inputs):
        x=self.tfc(inputs)

        inputs_proj=self.conv2d_res(inputs)
        return torch.add(x,inputs_proj)


class E_Block(nn.Module):

    def __init__(self, layer_idx,N0,N01, N,  S, num_tfc, **kwargs):
        super(E_Block, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.N0=N0
        self.N=N
        self.S=S
        self.i_block=I_Block(N0,N01,num_tfc)

        ksize=(S[0]+2,S[1]+2)
        self.conv2d_2 = nn.Sequential(nn.Conv2d(N01,N,
                                          kernel_size=(S[0]+2,S[1]+2),
                                          padding=(2,2),
                                          stride=S,
                                          padding_mode='reflect'),
                                      nn.ELU())


    def forward(self, inputs, training=None, **kwargs):
        x=self.i_block(inputs)
        
        x_down = self.conv2d_2(x)

        return x_down, x


class D_Block(nn.Module):

    def __init__(self, layer_idx,N0, N,  S,  num_tfc, **kwargs):
        super(D_Block, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.N=N
        self.S=S
        ksize=(S[0]+2, S[1]+2)
        self.tconv_1= nn.Sequential(
                                nn.ConvTranspose2d(N0,N,
                                             kernel_size=(S[0]+2, S[1]+2),
                                             stride=S,
                                             padding_mode='zeros'),
                                nn.ELU())

        self.upsampling = nn.Upsample(scale_factor=S, mode="nearest")

        self.projection =nn.Conv2d(N0,N,
                                      kernel_size=(1,1),
                                      stride=1,
                                      padding='same',
                                      padding_mode='reflect')
        self.cropadd=CropAddBlock()
        self.cropconcat=CropConcatBlock()

        self.i_block=I_Block(2*N,N,num_tfc)

    def forward(self, inputs, bridge, **kwargs):
        x = self.tconv_1(inputs)

        x2= self.upsampling(inputs)

        if x2.shape[-1]!=x.shape[-1]:
            x2= self.projection(x2)

        x= self.cropadd(x,x2)
        
        x=self.cropconcat(x,bridge)

        x=self.i_block(x)
        return x


class CropAddBlock(nn.Module):

    def forward(self,down_layer, x,  **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        #print(x1_shape,x2_shape)
        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2


        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff)]
        x = torch.add(down_layer_cropped, x)
        return x

class CropConcatBlock(nn.Module):

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2
        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff)]
        x = torch.cat((down_layer_cropped, x),1)
        return x
