import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    
    def forward(self, x, noise):
        enc_out = self.encoder(x)
<<<<<<< HEAD
=======
        
>>>>>>> TrainModule
        out = torch.cat((noise, enc_out), 1)
        
        out = self.decoder(out, self.encoder.skips)

        self.encoder.skips.clear()
        
        return out
<<<<<<< HEAD
    
=======
    
    
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
        win_gap = 2 << 13
        
        noisy_sample_segments = [noisy_waveform[i: i+win_len] for i in range(0, noisy_waveform.shape[0] - win_len, win_gap)]
        
        
        # zero pad the last segment
        if noisy_waveform.shape[0] % win_len > 0:
            last_segment = noisy_waveform[win_len * len(noisy_sample_segments): ]
            
            padding_len = win_len - len(last_segment)
            last_segment = np.pad(last_segment, (0, padding_len), mode='constant', constant_values = (0))
            
            noisy_sample_segments += [last_segment]

        #last_seg_len = len(noisy_sample_segments[-1])
        #noisy_sample_segments[-1] = np.pad(noisy_sample_segments[-1], win_len - last_seg_len)
        # last_offset = noisy_waveform.shape[0] // win_len
        # if noisy_waveform.shape[0] % win_len != 0:
        #     #res = noisy_waveform.shape[0] % win_len
            
            
        #     last_segment = np.zeros(win_len)
        #     non_zero_len = noisy_waveform.shape[0] - last_offset*win_len
        #     last_segment[0: non_zero_len] = noisy_waveform[last_offset*win_len:]
            
        #     #last_segment = np.pad(last_segment, win_len - last_segment.shape[0], mode='constant',  constant_values=(0))
            
        #     noisy_sample_segments += [last_segment]
        
        
        
        for segment in noisy_sample_segments:
            noise = torch.normal(0, 1, (1, 1024, 8)).to(device)

            # segment = (2./65535.) * (segment - 32767) + 1
            
            #segment = self.emphasis(np.reshape(segment, (1,1, segment.shape[-1])))[0,0,:]

            segment = torch.from_numpy(segment).type(torch.FloatTensor)
            
            segm_batch = segment.unsqueeze(0).unsqueeze(0).to(device)
            
            out = self.forward(segm_batch, noise).reshape(-1).to('cpu')
            
            #model_out = self.emphasis(np.reshape(out.detach().numpy(), (1,1, out.shape[-1])), pre=False)[0,0,:]
            
            
            #segment = (2./65535.) * (segment - 32767) + 1
            
            
            cleaned_segments.append(out.detach().numpy())

        n_audio_samples = np.hstack(cleaned_segments)
        #n_audio_samples =  n_audio_samples * (1 / n_audio_samples.max())
        #n_audio_samples = self._denormalize_wave_minmax(n_audio_samples)
        
    
        
        return n_audio_samples
>>>>>>> TrainModule
