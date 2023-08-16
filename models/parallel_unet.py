import torch
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
    
class CrossAttention(nn.Module):
    """
    Cross attention module as introduced in the paper.
    """
    def __init__(self, in_channels, heads=8):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.query = nn.Linear(in_channels, in_channels * heads)
        self.key = nn.Linear(in_channels, in_channels * heads)
        self.value = nn.Linear(in_channels, in_channels * heads)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(in_channels * heads, in_channels)
        
    def forward(self, x, context):
        # Create multi-head queries, keys, and values
        Q = self.query(x).view(x.size(0), -1, self.heads, x.size(-1)).permute(0, 2, 1, 3)
        K = self.key(context).view(context.size(0), -1, self.heads, context.size(-1)).permute(0, 2, 1, 3)
        V = self.value(context).view(context.size(0), -1, self.heads, context.size(-1)).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.einsum("ijkl,ijml->ijkm", [Q, K])
        attention_probs = self.softmax(scores)
        
        # Compute attended values
        attended_values = torch.einsum("ijkl,ijlm->ijkm", [attention_probs, V])
        
        # Combine heads and pass through output layer
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, x.size(-1))
        return self.output(attended_values)

class EnhancedParallelUNet(nn.Module):
    """
    Enhanced Parallel UNet architecture with cross attention.
    """
    def __init__(self, in_channels, out_channels):
        super(EnhancedParallelUNet, self).__init__()
        
        # Existing UNet blocks
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.dec1 = UNetBlock(256, 128)
        self.dec2 = UNetBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Cross attention blocks
        self.cross_attention1 = CrossAttention(64, heads=8)
        self.cross_attention2 = CrossAttention(128, heads=8)
        self.cross_attention3 = CrossAttention(256, heads=8)
        
        # Other utilities (pooling, upsampling) remain the same
        
    def forward(self, x, context):
        # Encoding path with cross attention
        e1 = self.enc1(x)
        e1_attended = self.cross_attention1(e1, context)
        e2 = self.enc2(self.pool(e1_attended))
        e2_attended = self.cross_attention2(e2, context)
        e3 = self.enc3(self.pool(e2_attended))
        e3_attended = self.cross_attention3(e3, context)
        
        # Decoding path remains the same
        d1 = self.dec1(self.upsample(e3_attended))
        d2 = self.dec2(self.upsample(d1))
        out = self.final_conv(d2)
        
        return out
