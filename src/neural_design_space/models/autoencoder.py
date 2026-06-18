import torch
import torch.nn as nn
from torchinfo import summary
from neural_design_space.utils.utils import kernel_initializer




class AutoEncoderStem(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=latent_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.pointwise_conv = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.LazyConvTranspose2d(out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConvTranspose2d(out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.pointwise_conv(x)
        x = x.view(x.size(0), 64, 1, 1)
        x = self.decoder(x)
        return x