import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    """
    Basic block for UNet architecture.
    """
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class ParallelUNet(nn.Module):
    """
    Parallel UNet architecture as introduced in the paper.
    """
    def __init__(self, in_channels, out_channels):
        super(ParallelUNet, self).__init__()
        
        # Encoding layers
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        
        # Decoding layers
        self.dec1 = UNetBlock(256, 128)
        self.dec2 = UNetBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoding path
        d1 = self.dec1(self.upsample(e3))
        d2 = self.dec2(self.upsample(d1))
        
        # Final layer
        out = self.final_conv(d2)
        
        return out

# ... Cross attention and other layers as per paper would be added here ...
