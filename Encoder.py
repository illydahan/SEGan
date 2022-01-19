import torch
import torch.nn as nn
from BuildingBlocks import ConvLayer

#######################
#      Encoder        #
#######################


conv_filters_enc = [16, 32, 32, 64, 64, 128, 128 ,256 ,256, 512, 1024]

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