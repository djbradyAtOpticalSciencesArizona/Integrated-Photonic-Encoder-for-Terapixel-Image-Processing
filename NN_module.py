import torch 
import torch.nn as nn
from torch.nn import functional as F

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride, bias=False
        )

    def forward(self, x):
        return self.upsample(x)
    
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, kernel_sizes=3, paddings=1, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=kernel_sizes, padding=paddings, stride=strides, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=kernel_sizes, padding=paddings, bias=False)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides, bias=False)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.leaky_relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.leaky_relu(Y)