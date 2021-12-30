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
        out = torch.cat((noise, enc_out), 1)
        
        out = self.decoder(out, self.encoder.skips)

        self.encoder.skips.clear()
        
        return out
    