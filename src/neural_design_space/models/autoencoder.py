import torch
import torch.nn as nn
from torchinfo import summary
from neural_design_space.utils.utils import kernel_initializer




class AutoEncoderStem(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(out_features=latent_dim)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x