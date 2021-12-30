import torch
import torch.nn as nn
from WSEGan import TranConvLayer




conv_filters_dec = [(2048, 512), (1024, 256), (512, 256), (512, 128), (256, 128), (256, 64), (128, 64), (128, 32), (64, 32), (64, 16) ,(32, 1)]

#######################
#      Decoder        #
#######################

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for (in_dim, out_dim) in conv_filters_dec:

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

        return x