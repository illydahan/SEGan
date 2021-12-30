import os
import torch
from torch.utils.data import Dataset
import torchaudio
from scipy.io.wavfile import read
from tqdm import tqdm

class SoundDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir,reload=False ,transform=None):
        super().__init__()
        
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.sample_len = 16384 
        self.n_files = 0
        
        self.clean_samples = []
        self.noisy_samples = []
        
        if reload:
            self._reload_data()
        else:
            self._read_from_file()
    
    def _read_from_file(self):
        self.clean_samples = torch.load('sound_data/clean.pt').tolist()
        self.noisy_samples = torch.load('sound_data/noisy.pt').tolist()
        
        print ("samples shape: ")
        print(self.noisy_samples[0].shape)
        
        print(f"loaded : {len(self.clean_samples)} samples")
        
    
            
    
    def _reload_data(self):
        
        # list of all the files names
        clean_files = os.listdir(self.clean_dir)
        noisy_files = os.listdir(self.noisy_dir)
        
        self.n_files = len(clean_files)
        
        assert len(clean_files) == len(noisy_files)
        
        samples_gap = 8000
        
        for (fn_clean, fn_noisy) in tqdm(zip(clean_files, noisy_files)):
            s1, waveform_c = read(os.path.join(self.clean_dir, fn_clean))
            s2, waveform_n = read(os.path.join(self.noisy_dir, fn_noisy))
            
            if s1 != 16e3 or s2 != 16e3:
                continue
            
            
            # waveform_c = waveform_c.type(torch.FloatTensor)
            # waveform_n = waveform_n.type(torch.FloatTensor)
            
            # pre emphasis high frequency
            
            #waveform_c = emphasis(np.reshape(waveform_c, (1,1,waveform_c.shape[-1])))[0,0,:]
            #waveform_n = emphasis(np.reshape(waveform_n, (1,1,waveform_n.shape[-1])))[0,0,:]
            
            # waveform_c = (2./65535.) * (waveform_c.astype(np.float32) - 32767.) + 1.
            # waveform_n = (2./65535.) * (waveform_n.astype(np.float32) - 32767.) + 1.
            
            
            waveform_c = torch.from_numpy(waveform_c).type(torch.FloatTensor)
            waveform_n = torch.from_numpy(waveform_n).type(torch.FloatTensor)
            
            waveform_c.squeeze_(0)
            waveform_n.squeeze_(0)
            
            
            
            self.clean_samples += [waveform_c[i:i+self.sample_len] for i in range(0, waveform_c.shape[0] - self.sample_len, samples_gap)]
            self.noisy_samples += [waveform_n[i:i+self.sample_len] for i in range(0, waveform_n.shape[0] - self.sample_len, samples_gap)]
                    
        
    def __len__(self):
        return self.n_files
    
    def __getitem__(self, index):
        return (self.clean_samples[index], self.noisy_samples[index])
    
    

import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import librosa



def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1,1)
    axs.set_title(title or 'Spectogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    plt.waitforbuttonpress()
    

def plot_spectrogram_noisy_clean(spec: list, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1,2)
    axs[0].set_title(title or 'Spectogram (db) clean')
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel('frame')
    
    axs[1].set_title(title or 'Spectogram (db) noisy')
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlabel('frame')
    
    im_clean = axs[0].imshow(librosa.power_to_db(spec[0]), origin='lower', aspect=aspect)
    im_noisy = axs[1].imshow(librosa.power_to_db(spec[1]), origin='lower', aspect=aspect)
    if xmax:
        axs[0].set_xlim((0, xmax))
        axs[1].set_xlim((0, xmax))
        
    fig.colorbar(im_clean, ax=axs)
    plt.show(block=False)
    plt.waitforbuttonpress()
    
fs, test_sample = read('sound_data/clean/p226_008.wav')
_, noisy_sample = read('sound_data/noisy/p226_008.wav')

test_sample = torch.from_numpy(test_sample.astype(np.float32))
noisy_sample = torch.from_numpy(noisy_sample.astype(np.float32))


stft_sig = torch.stft(test_sample.type(torch.float32), fs // 2, onesided=True)

win_len = 2 << 13
hop = int(8e3)
to_specto = torchaudio.transforms.Spectrogram(n_fft = test_sample.shape[0], win_length=win_len, hop_length=hop)



import math
spect_signal = to_specto(test_sample)
noisy_sample = to_specto(noisy_sample)

padded_sig_len = 2 ** math.ceil(math.log2(test_sample.shape[0] // 2  + 1))

dummy_signal = torch.zeros(padded_sig_len)
n_win = len([dummy_signal[i:i+win_len] for i in range(0, padded_sig_len, hop)])
print(f"signal composed of {n_win} segments")


plot_spectrogram_noisy_clean([spect_signal, noisy_sample])

x = 1