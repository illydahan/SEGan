from generator import Generator
from scipy.io.wavfile import read, write

import matplotlib.pyplot as plt
import numpy as np
import torch

import os
from pesq import pesq

from scipy.fft import fft, fftfreq
import scipy.signal as sps

import argparse


def merge_generated(gen_signal, audio_len= 0):
    window_len = 2 << 13
    sample_gap = window_len // 2
    
    
    n_windows = audio_len // window_len
    
    
    windows = [gen_signal[i:i+window_len] for i in range(0, gen_signal.shape[0] - window_len, sample_gap)]
    
    merged_windows = []
    
    start_index = 0
    
    for i in range(0, len(windows), 4):
        merged_windows.append(windows[i])
        
        
    merged_windows = np.hstack(merged_windows)
    
    return merged_windows
        
        
    

def normalize(x):
    return (2./65535.) * (x - 32767) + 1

# def merge_overlape(x):
#     sample_len = 16384
#     x_seg = [x[i:i+sample_len] for i in range(0, x.shape[-1] - sample_len, 8000)]
    
#     merged_sig = []
#     offset = 0
#     for i in range(len(x_seg) - 1):
#         new_sig = np.hstack(x_seg[i][offset:8000])









def main():
    parser = argparse.ArgumentParser(description='Clean a noisy waveform with segan')
    parser.add_argument('wav file noisy input', )









if __name__ == '__main__':
    main()
    









gen = Generator().eval()
gen.load_state_dict(torch.load(r'checkpoints/gen_final.pth'))


noisy_sample = 'test_hebrew.wav'
#noisy_sample = 'sound_data/noisy/p226_028.wav'
clean_sample = 'sound_data/clean/p226_028.wav'


_, clean_waveform = read(clean_sample)
sample_rate, noisy_waveform = read(noisy_sample)



if sample_rate != 16e3:
    number_of_samples = round(len(noisy_waveform) * float(16e3) / sample_rate)
    noisy_waveform = sps.resample(noisy_waveform, number_of_samples)
    
    print(f"Audio downsampled from : {sample_rate} to 16k")

    sample_rate = int(16e3)
    
#cleaned_signal = gen2.clean(noisy_waveform)
cleaned_signal = gen.clean(noisy_waveform[:,0])

#cleaned_signal = merge_generated(cleaned_signal)

import scipy
#cleaned_signal = merge_generated(cleaned_signal)

# cleaned_signal = np.clip(cleaned_signal, -1, 1)

# clean_waveform = normalize(clean_waveform)
val = pesq(sample_rate, clean_waveform.astype(np.int16), cleaned_signal.astype(np.int16), 'nb')
print(f"pesq: {val}")



fix, axes = plt.subplots(2, 3)
write("test.wav", sample_rate, cleaned_signal.astype(np.int16))

N = cleaned_signal.shape[0]
sample_rate = 16e3

# Time domain - clean
t_clean = np.arange(0, clean_waveform.shape[0] * 1/sample_rate, 1/sample_rate)
axes[0, 0].plot(t_clean, clean_waveform.astype(np.float32))
axes[0, 0].set_title("Clean Signal")

Ts_clean = t_clean[1] - t_clean[0]
clean_freq = fft(clean_waveform.astype(np.float32)) 
clean_freq_range = fftfreq(clean_waveform.shape[0] , Ts_clean)[:N // 2]

# Frequency doimain - clean
axes[1, 0].plot(clean_freq_range, (2/N) * np.abs(clean_freq[:N//2]) )
axes[1, 0].set_title("Clean Frequency")


# Time domain - noisy
t_noisy = np.arange(0, noisy_waveform.shape[0]  * 1/sample_rate, 1/sample_rate)
axes[0, 1].plot(t_noisy, noisy_waveform.astype(np.float32))
axes[0, 1].set_title("Noisy Signal")


# Frequency domain - noisy
Ts_noisy = t_noisy[1] - t_noisy[0]
noisy_freq = fft(noisy_waveform.astype(np.float32))[:N//2]
noisy_freq_range = fftfreq(noisy_waveform.shape[0] , Ts_noisy)[:N // 2]

axes[1, 1].plot(noisy_freq_range, (2/N) * np.abs(noisy_freq) )
axes[1, 1].set_title("Noisy Frequency")

# Time domain - generated signal
t_gen = np.arange(0, cleaned_signal.shape[0] * 1/sample_rate, 1/sample_rate)
axes[0, 2].plot(t_gen, cleaned_signal.astype(np.float32))
axes[0, 2].set_title("Generated Signal")

# Frequency doimain - generated signal
Ts_gen = t_gen[1] - t_gen[0]
gen_freq = fft(cleaned_signal.astype(np.float32))[:N//2]
gen_freq_range = fftfreq(cleaned_signal.shape[0], Ts_gen)[:N // 2]
axes[1, 2].plot(gen_freq_range, (2/N) * np.abs(gen_freq) )
axes[1, 2].set_title("Generated Frequency")



#plt.plot(np.arange(clean_waveform.shape[0]), clean_waveform.astype(np.float32))
write("original.wav", 16000, clean_waveform.astype(np.int16))
plt.show()
