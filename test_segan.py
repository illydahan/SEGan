import enum
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

from utils import SSNR


def plot_frequency_response(gen:Generator, clean_waveform, device='cuda'):
    isnr = np.array([20, 5, 0]) # isnr in db range
    rx = np.mean(clean_waveform ** 2)
    
    N = clean_waveform.shape[0]
    fs = int(16e3)
    freq = np.arange(0, fs//2, fs / N)
    
    h = []
    for ii, i_snr in enumerate(isnr):
        isnr_linear = (10 ** (i_snr / 20))
        rn = rx / isnr_linear
        
        noise = np.random.normal(0, int(rn ** 0.5), N)
        
        noisy_signal = clean_waveform + noise
        
        noisy_freq = fft(noisy_signal)[:N//2]
        
        generated_signal = gen.clean(noisy_signal, device)
        
        gen_freq = fft(generated_signal)[:N//2]
        
        h += [np.abs(gen_freq) / np.abs(noisy_freq)]
        
    
    fig, axes = plt.subplots(nrows=1, ncols=3)
    
    
    axes[0].plot(freq, h[0])
    axes[0].set_ylabel('H')
    axes[0].set_xlabel('f [HZ]')
    axes[0].set_title("Frequency Response for iSNR = 20[dB]")
    
    axes[1].plot(freq, h[1])
    axes[1].set_ylabel('H')
    axes[1].set_xlabel('f [HZ]')
    axes[1].set_title("Frequency Response for iSNR = 5[dB]")
    
    axes[2].plot(freq, h[2])
    axes[2].set_ylabel('H')
    axes[2].set_xlabel('f [HZ]')
    axes[2].set_title("Frequency Response for iSNR = 1[dB]")
    plt.show()

    
    
def plot_results_normal_noise(gen: Generator, clean_waveform, device='cuda'):
    """
    Plot SSNR, and MSE for normal noise

    Args:
        gen (Generator): [Generatir]
        clean_waveform ([type]): [Clean Signal]
    """
    
    isnr = np.arange(21) # isnr in db range
    
    mse_vec = np.zeros_like(isnr, dtype=np.float32)
    ssnr_vec = np.zeros_like(isnr, dtype=np.float32)
    
    rx = np.mean(clean_waveform ** 2)
    for ii, i_snr in enumerate(isnr):
        
        isnr_linear = (10 ** (i_snr / 20))
        
        
        
        rn = rx / isnr_linear
        
        noise = np.random.normal(0, int(rn ** 0.5), clean_waveform.shape[0])
        
        noisy_signal = clean_waveform + noise
        
        generated_signal = gen.clean(noisy_signal, device)
        
        mse = np.mean((clean_waveform - generated_signal) ** 2)
        
        ssnr = SSNR(clean_waveform, generated_signal)
        
        mse_vec[ii] = 20 * np.log10(mse)
        ssnr_vec[ii] = 20*np.log10(ssnr)
        

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(isnr, mse_vec)
    axes[0].set_xlabel('iSNR [dB]')
    axes[0].set_ylabel('MSE [dB]')
    
    axes[1].plot(isnr, ssnr_vec)
    axes[1].set_xlabel('iSNR[dB]')
    axes[1].set_ylabel('SSNR [dB]')
    plt.show()

def model_frequency_response(gen_signal_freq, input_signal_freq):
    """
        Return the frequency response of the model (Transfer function)
    Args:
        gen_signal_freq ([type]): [description]
        input_signal_freq ([type]): [description]
    """
    
    assert gen_signal_freq.shape == input_signal_freq.shape
    
    freq_response = np.abs(gen_signal_freq) / np.abs(input_signal_freq)
    
    return freq_response
    
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
        noisy_waveform = noisy_waveform[..., 1]
        
    generated_signal = generator.clean(noisy_waveform, device=device)
    
    #generated_signal = merge_generated(generated_signal)
    
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
    noisy_freq = fft(noisy_waveform.astype(np.float32))[:N // 2]
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
        noisy_waveform += np.random.normal(mu, sigma, (noisy_waveform.shape[0])).astype(noisy_waveform.dtype)

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
    
    clean_file = "sound_data/clean/p226_007.wav"
    noisy_file = "sound_data/noisy/p226_007.wav"
    
    _,noisy_waveform = read(noisy_file)
    _,clean_waveform = read(clean_file)
    
    
    weights_file = "checkpoints/generator_parms_pesq.pth"
    out_file = "out.wav"

    args = parser.parse_args()
    
    max_val = ((2 << 15) / 2) - 1
    sigma = max_val * 0.03
    #noisy_waveform, clean_waveform = make_noisy_sample(in_file, mu = 0, sigma = sigma)
   
    gen = Generator().eval()
    gen.load_state_dict(torch.load(weights_file))
    
    
    #plot_frequency_response(gen, clean_waveform, device='cpu')
    #plot_results_normal_noise(gen, clean_waveform, device='cpu')
    clean_noisy(noisy_waveform, gen, out_file, clean_waveform=clean_waveform)
    

if __name__ == '__main__':
    
    main()
    





