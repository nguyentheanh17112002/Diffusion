import torch
import torch.nn as nn
import torch.nn.functional as F

from .DoubleConv import DoubleConv  
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

    
if __name__ == "__main__":
    x = torch.rand((4,64,572,572))
    print(x.shape)
    conv = Down( in_channels=64, out_channels=128)
    out = conv(x)
    print(out.shape)