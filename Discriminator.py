import torch
import torch.nn as nn
from BuildingBlocks import ConvLayer

disc_layers = [(2, 32), (32, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 512), (512, 512), (512, 1024), (1024, 2048)]



class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.mainModel = nn.ModuleList()
        self.leakyRelu =  nn.LeakyReLU(0.3)
        self.last_conv = ConvLayer(2048, 1, kernel_size=1, stride=1, use_bn=True, padding=0)
        self.linear = nn.Linear(in_features=8, out_features=1)
        self._init_layers()
        
    def forward(self, x):
        for layer in self.mainModel:
            x = layer(x)

        
        x = self.leakyRelu(self.last_conv(x))
                
        out = self.linear(x)
        
        return out.view(-1)
    
    def _init_layers(self):
        for layer in disc_layers:
            in_dim, out_dim = layer
            self.mainModel += [ConvLayer(in_dim, out_dim, gen=False)]