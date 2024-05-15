
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LinkNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LinkNetEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu2(out)
        return out

class LinkNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinkNetDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        return out

class MultiScaleLinkNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(MultiScaleLinkNet, self).__init__()

        self.encoder1 = LinkNetEncoderBlock(in_channels, 64, stride=1)
        self.encoder2 = LinkNetEncoderBlock(64, 128, stride=2)
        self.encoder3 = LinkNetEncoderBlock(128, 256, stride=2)
        self.encoder4 = LinkNetEncoderBlock(256, 512, stride=2)

        self.decoder4 = LinkNetDecoderBlock(512, 256)
        self.decoder3 = LinkNetDecoderBlock(256, 128)
        self.decoder2 = LinkNetDecoderBlock(128, 64)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid() 
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        print(f"Input Tensor Shape: {x.shape}")
        
        #Encoder
        enc1 = self.encoder1(x)
        print(f"Encoder 1 Output Shape: {enc1.shape}")
        
        enc2 = self.encoder2(enc1)
        print(f"Encoder 2 Output Shape: {enc2.shape}")

        enc3 = self.encoder3(enc2)
        print(f"Encoder 3 Output Shape: {enc3.shape}")
        
        enc4 = self.encoder4(enc3)
        print(f"Encoder 4 Output Shape: {enc4.shape}")

        #Decoder with skip connections
        dec4 = self.decoder4(enc4)
        dec3 = self.decoder3(dec4 + enc3)
        dec2 = self.decoder2(dec3 + enc2)
        dec1 = self.decoder1(dec2 + enc1)

        return dec1