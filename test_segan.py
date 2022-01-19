from Generator import Generator
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
    sample_gap = window_len
    
    windows = [gen_signal[i:i+window_len] for i in range(0, gen_signal.shape[0] - window_len, sample_gap)]
    
    #merged_windows = []

    merged_windows = np.hstack(windows)
    
    return merged_windows
        
        

def normalize(x):
    return (2./65535.) * (x - 32767) + 1


def clean_noisy(noisy_waveform, generator: Generator, out_file=None, device='cpu',  clean_waveform=None, sample_rate = int(16e3)):
    
    generator = generator.to(device)

    
    if len(noisy_waveform.shape) > 1:
        noisy_waveform = noisy_waveform[..., 0]
        
    generated_signal = generator.clean(noisy_waveform, device=device)
    
    generated_signal = merge_generated(generated_signal)
    
    if out_file is not None:
        write(out_file, sample_rate, generated_signal.astype(np.int16))


    try:
        val = pesq(sample_rate, noisy_waveform.astype(np.int16), generated_signal.astype(np.int16), 'nb')
        print(f"pesq: {val}")
    except:
        pass


    # ------------ #
    # Plot results #
    # ------------ #
    
    if clean_waveform is not None:
        fig, axes = plt.subplots(2, 3)
    else:
        fig, axes = plt.subplots(2, 2)

    N = generated_signal.shape[0]
    sample_rate = 16e3

    fig_idx = 0
    if clean_waveform is not None:
        # Time domain - clean
        t_clean = np.arange(0, clean_waveform.shape[0] * 1/sample_rate, 1/sample_rate)
        axes[0, fig_idx].plot(t_clean, clean_waveform.astype(np.float32))
        axes[0, fig_idx].set_title("Clean Signal")

        # Frequency doimain - clean
        Ts_clean = t_clean[1] - t_clean[0]
        clean_freq = fft(clean_waveform.astype(np.float32))[:N // 2]
        clean_freq_range = fftfreq(clean_waveform.shape[0] , Ts_clean)[:N // 2]
        axes[1, fig_idx].plot(clean_freq_range, (2/N) * np.abs(clean_freq) )
        axes[1, fig_idx].set_title("Clean Frequency")
        
        fig_idx += 1


    # Time domain - noisy
    t_noisy = np.arange(0, noisy_waveform.shape[0]  * 1/sample_rate, 1/sample_rate)
    axes[0, fig_idx].plot(t_noisy, noisy_waveform.astype(np.float32))
    axes[0, fig_idx].set_title("Noisy Signal")


    # Frequency domain - noisy
    Ts_noisy = t_noisy[1] - t_noisy[0]
    noisy_freq = fft(noisy_waveform.astype(np.float32))[:N//2]
    noisy_freq_range = fftfreq(noisy_waveform.shape[0] , Ts_noisy)[:N // 2]

    axes[1, fig_idx].plot(noisy_freq_range, (2/N) * np.abs(noisy_freq) )
    axes[1, fig_idx].set_title("Noisy Frequency")

    fig_idx += 1
    
    # Time domain - generated signal
    t_gen = np.arange(0, generated_signal.shape[0] * 1/sample_rate, 1/sample_rate)
    axes[0, fig_idx].plot(t_gen, generated_signal.astype(np.float32))
    axes[0, fig_idx].set_title("Generated Signal")

    # Frequency doimain - generated signal
    Ts_gen = t_gen[1] - t_gen[0]
    gen_freq = fft(generated_signal.astype(np.float32))[:N//2]
    gen_freq_range = fftfreq(generated_signal.shape[0], Ts_gen)[:N // 2]
    axes[1, fig_idx].plot(gen_freq_range, (2/N) * np.abs(gen_freq) )
    axes[1, fig_idx].set_title("Generated Frequency")

    plt.show()

def make_noisy_sample(noisy_path, mu = None, sigma=None):
    noisy_sample_rate, noisy_waveform = read(noisy_path)
    
    if len(noisy_waveform.shape) > 1:
        noisy_waveform = noisy_waveform[..., 0]
        
    if noisy_waveform.max() <= 1:
        noisy_waveform *= ((2 << 15) / 2 -1)
    
    clean_waveform = noisy_waveform.copy()
    if noisy_sample_rate != 16e3:
        number_of_samples = round(len(noisy_waveform) * float(16e3) / noisy_sample_rate)
        noisy_waveform = sps.resample(noisy_waveform, number_of_samples)

        print(f"Audio downsampled from : {noisy_sample_rate}[Hz] to 16[KHz]")
    
    
        
    if mu is not None and sigma is not None:
        noisy_waveform += np.random.normal(mu, sigma, (noisy_waveform.shape[0]))

    return noisy_waveform, clean_waveform

def main():
    
    parser = argparse.ArgumentParser(description='Clean a noisy waveform with segan')
    # parser.add_argument('--in_file', help = 'wav file noisy input', type=str, required=True)
    # parser.add_argument('--out_file', help='cleaned wav output file', type=str, required=False)
    # parser.add_argument('--device', help='cuda or cpu, to run the model on. (defaults to cpu)', type=str, required=False, default='cpu')
    # parser.add_argument('--weights', help='Saved weight for the generator', type=str, required=False, default='checkpoints/gen_final.pth')
    
    # in_file = args.in_file
    # device = args.device
    # out_file = args.out_file
    
    # weights_file = args.weights
    
    in_file = "test_samples/test_clean.wav"
    weights_file = "checkpoints/generator_parms_ssnr.pth"
    out_file = "out.wav"

    args = parser.parse_args()
    
    max_val = ((2 << 15) / 2) - 1
    sigma = max_val * 0.1 
    noisy_waveform, clean_waveform = make_noisy_sample(in_file, mu = 0, sigma = sigma)
   
    gen = Generator().eval()
    gen.load_state_dict(torch.load(weights_file))
    
    
    clean_noisy(noisy_waveform, gen, out_file, clean_waveform=clean_waveform)
    

if __name__ == '__main__':
    main()
    





