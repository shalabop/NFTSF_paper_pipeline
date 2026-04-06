import torch
from torch import nn
from ...TCN.tcn import TemporalConvNet
import torch.nn.functional as F

class TimeSeriesTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TimeSeriesTCN, self).__init__()
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    
    def encode(self, x):
        y = self.tcn(x)
        z = y[:, :, -1]
        return F.normalize(z, dim=1)

    def forward(self, x):
        """
        Used for standard training.
        """

        if x.shape[1] != 1: 
            x = x.transpose(1, 2)
            
        feat = self.encode(x)
        return self.decoder(feat)
