import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from scipy import signal


import torchaudio

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

# audioIO
from scipy.io.wavfile import write
from scipy.io.wavfile import read

# play audio
import sounddevice as sd

class ConvLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True, kernel_size=31, stride = 2, padding = 31 // 2, gen=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride = stride, padding=padding, padding_mode='reflect')
        self.parmRelu = nn.PReLU()
        self.leakyRelu = nn.LeakyReLU(0.3)
        self.gen = gen
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        if self.gen:
            return self.parmRelu(self.conv(x))
        else:
            return self.leakyRelu(self.bn(self.conv(x)))
        
        
class TranConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_features, out_features, kernel_size=32, stride = 2, padding= 31 // 2)
        self.parmRelu = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        return self.parmRelu(self.conv(x))
    

conv_filters_enc = [16, 32, 32, 64, 64, 128, 128 ,256 ,256, 512, 1024]
conv_filters_dec = [(2048, 512), (1024, 256), (512, 256), (512, 128), (256, 128), (256, 64), (128, 64), (128, 32), (64, 32), (64, 16) ,(32, 1)]

class Encoder(nn.Module):
    def __init__(self, in_dim=1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.skips = []
        current_dim = in_dim
        for fil in conv_filters_enc:
            self.layers += [ConvLayer(current_dim, fil)]
            current_dim = fil
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            self.skips.append(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for (in_dim, out_dim) in conv_filters_dec:
            # if dim == 1:
            #     break
            
            self.layers += [TranConvLayer(in_dim, out_dim)]

        print(len(self.layers))
        

    def forward(self, x, skips):
        # ignore first skip
        skips.pop()
        for idx, layer in enumerate(self.layers):
            if idx < 10:
                skip = skips.pop()
                x = torch.cat((layer(x), skip), 1)
            else:
                x = layer(x)
                
            #print(x.shape)
        
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    
    def forward(self, x, noise):
        enc_out = self.encoder(x)
        
        #noise = torch.normal(0 , 1, (enc_out.shape[0], 1024, 8)).to(device)
        out = torch.cat((noise, enc_out), 1)
        
        out = self.decoder(out, self.encoder.skips)
        
        
        self.encoder.skips.clear()
        
        return out
    
    
    def _denormalize_wave_minmax(self, x):
        return (65535. * x / 2) - 1 + 32767.
    
    def emphasis(self, signal_batch, emph_coeff=0.95, pre=True):
        """
        Pre-emphasis or De-emphasis of higher frequencies given a batch of signal.
        Args:
            signal_batch: batch of signals, represented as numpy arrays
            emph_coeff: emphasis coefficient
            pre: pre-emphasis or de-emphasis signals
        Returns:
            result: pre-emphasized or de-emphasized signal batch
        """
        result = np.zeros(signal_batch.shape)
        for sample_idx, sample in enumerate(signal_batch):
            for ch, channel_data in enumerate(sample):
                if pre:
                    result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
                else:
                    result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
        return result

    def clean(self, noisy_waveform, device='cpu'):
        cleaned_segments = []
        win_len = 2 << 13
        
        win_gap = (2 << 13)
        noisy_sample_segments = [noisy_waveform[i:i+win_len] for i in range(0, noisy_waveform.shape[0] - win_len, win_gap)]

        
        if noisy_waveform.shape[0] % win_len != 0:
            #res = noisy_waveform.shape[0] % win_len
            last_offset = noisy_waveform.shape[0] // win_len
            
            last_segment = np.zeros(win_len)
            non_zero_len = noisy_waveform.shape[0] - last_offset*win_len
            last_segment[0: non_zero_len] = noisy_waveform[last_offset*win_len:]
            
            #last_segment = np.pad(last_segment, win_len - last_segment.shape[0], mode='constant',  constant_values=(0))
            
            noisy_sample_segments += [last_segment]
        
        
        
        for segment in noisy_sample_segments:
            noise = torch.normal(0, 1, (1, 1024, 8)).to(device)

            # segment = (2./65535.) * (segment - 32767) + 1
            
            #segment = self.emphasis(np.reshape(segment, (1,1, segment.shape[-1])))[0,0,:]
            

            
            segment = torch.from_numpy(segment).type(torch.FloatTensor)
            
            segm_batch = segment.unsqueeze(0).unsqueeze(0).to(device)
            
            out = self.forward(segm_batch, noise)[0,0,:].to('cpu')
            
            #model_out = self.emphasis(np.reshape(out.detach().numpy(), (1,1, out.shape[-1])), pre=False)[0,0,:]
            
            
            #segment = (2./65535.) * (segment - 32767) + 1
            
            
            cleaned_segments.append(out.detach().numpy())

        n_audio_samples = np.hstack(cleaned_segments)
        #n_audio_samples =  n_audio_samples * (1 / n_audio_samples.max())
        #n_audio_samples = self._denormalize_wave_minmax(n_audio_samples)
        return n_audio_samples