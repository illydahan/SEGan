# from subprocess import run, PIPE
# from scipy.linalg import toeplitz
# from scipy.io import wavfile
# import numba as nb
# from numba import jit, int32, float32
# import soundfile as sf
# from scipy.signal import lfilter
# from scipy.interpolate import interp1d

# import torch
# import numpy as np

# from pesq import pesq

# def SSNR(ref, deg, M=1000):
#     sig_len = ref.shape[-1]
#     ssnr = 0
#     ref_chopped = [ref[i:i+M] for i in range(0, sig_len - M, M)]
#     deg_chopped = [deg[i:i+M] for i in range(0, sig_len - M, M)]
    
#     for ref_seg, deg_seg in zip(ref_chopped, deg_chopped):
#         segment_snr = np.sum(ref_seg ** 2) / (np.sum((ref_seg - deg_seg) ** 2) + 1e-3)
#         ssnr += segment_snr
    
#     ssnr = (10 / M) * ssnr
    
#     return ssnr

# def normalize_sound(signal):
#     pass


# def evalutate_batch_performance(gen, sound_data, sound_loader, device):
#     random_idx = np.random.randint(0, len(sound_loader), size=(300))
    
#     running_pesq_avg = 0
#     running_ssnr_avg = 0
#     error = 0
    
#     n_valid_clean = 0
#     for idx in random_idx:
        
#         clean_data = sound_data.clean_samples[idx]
#         noisy_data = sound_data.noisy_samples[idx]
#         if clean_data.shape[-1] != 16384 or noisy_data.shape[-1] != 16384:
#             continue
        
#         z = torch.normal(0, 1, (1, 1024, 8)).to(device)
#         gen_fake_sample = gen(torch.reshape(noisy_data, (1,1, 16384)).to(device), z).to('cpu')
        
#         gen_fake_sample = gen_fake_sample.view(16384).detach().numpy()
        
#         running_ssnr_avg += SSNR(clean_data.numpy(), gen_fake_sample)
#         try:
#             rescaled_signal =  (clean_data.numpy())
#             running_pesq_avg += pesq(16000, clean_data.numpy(), gen_fake_sample, 'wb')
#             n_valid_clean += 1
#         except:
#             error = 1
#             break                   
                                
            
#     if n_valid_clean < 0:
#         curr_pesq = running_pesq_avg / n_valid_clean
#     else:
#         curr_pesq = 0
        
#     curr_ssnr = running_ssnr_avg / len(sound_loader)
    
    
import torch
import numpy as np

from pesq import pesq

def SSNR(ref, deg, M=1000):
    sig_len = ref.shape[-1]
    ssnr = 0
    ref_chopped = [ref[i:i+M] for i in range(0, sig_len - M, M)]
    deg_chopped = [deg[i:i+M] for i in range(0, sig_len - M, M)]
    
    for ref_seg, deg_seg in zip(ref_chopped, deg_chopped):
        segment_snr = np.sum(ref_seg ** 2) / (np.sum((ref_seg - deg_seg) ** 2) + 1e-3)
        ssnr += segment_snr
    
    ssnr = (10 / M) * ssnr
    
    return ssnr

def normalize_sound(signal):
    pass


def evalutate_batch_performance(gen, sound_data, sound_loader, device):
    random_idx = np.random.randint(0, len(sound_loader), size=(300))
    
    running_pesq_avg = 0
    running_ssnr_avg = 0
    error = 0
    
    n_valid_clean = 0
    for idx in random_idx:
        
        clean_data = sound_data.clean_samples[idx]
        noisy_data = sound_data.noisy_samples[idx]
        if clean_data.shape[-1] != 16384 or noisy_data.shape[-1] != 16384:
            continue
        
        z = torch.normal(0, 1, (1, 1024, 8)).to(device)
        gen_fake_sample = gen(torch.reshape(noisy_data, (1,1, 16384)).to(device), z).to('cpu')
        
        gen_fake_sample = gen_fake_sample.view(16384).detach().numpy()
        
        running_ssnr_avg += SSNR(clean_data.numpy(), gen_fake_sample)
        try:
            rescaled_signal =  (clean_data.numpy())
            running_pesq_avg += pesq(16000, clean_data.numpy(), gen_fake_sample, 'wb')
            n_valid_clean += 1
        except:
            error = 1
            break                   
                                
            
    if n_valid_clean < 0:
        curr_pesq = running_pesq_avg / n_valid_clean
    else:
        curr_pesq = 0
        
    curr_ssnr = running_ssnr_avg / len(sound_loader)
    
    
    return curr_ssnr, curr_pesq
