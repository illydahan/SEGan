import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True, kernel_size=31, stride = 2, padding = 31 // 2, gen=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride = stride, padding=padding, padding_mode='reflect')
        self.parmRelu = nn.PReLU()
        self.leakyRelu = nn.LeakyReLU(0.3)
        self.gen = gen
        self.init_weights()
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        if self.gen:
            return self.parmRelu(self.conv(x))
        else:
            return self.leakyRelu(self.bn(self.conv(x)))
    
    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

class TranConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_features, out_features, kernel_size=32, stride = 2, padding= 31 // 2)
        self.parmRelu = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_features)
        self.init_weights()
    
    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)
        
    def forward(self, x):
        return self.parmRelu(self.conv(x))