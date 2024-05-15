import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import resnet18

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

        # Initialize convolutional layers
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

        # Use a pretrained ResNet-18 as the encoder
        self.encoder = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove last two layers (avgpool and fc)

        # Decoder (upsampling)
        self.decoder4 = LinkNetDecoderBlock(512, 256)
        self.decoder3 = LinkNetDecoderBlock(256 + 256, 128)
        self.decoder2 = LinkNetDecoderBlock(128 + 128, 64)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 + 64, num_classes, kernel_size=1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

        # Classifier layers after the decoder
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.tp_conv2 = nn.ConvTranspose2d(32, num_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)

        # Initialize convolutional layers in the decoders
        self._init_conv_weights()

    def _init_conv_weights(self):
        for m in self.decoder4.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.decoder3.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.decoder2.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.decoder1.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, x):
        # Encoder
        enc = self.encoder(x)

        # Decoder with skip connections
        dec4 = self.decoder4(enc)
        dec3 = self.decoder3(torch.cat((dec4, enc), dim=1))
        dec2 = self.decoder2(torch.cat((dec3, enc), dim=1))
        dec1 = self.decoder1(torch.cat((dec2, enc), dim=1))

        # Classifier
        y = self.tp_conv1(dec1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        y = self.lsm(y)

        return y

# Instantiate and use the MultiScaleLinkNet model
model = MultiScaleLinkNet(in_channels=3, num_classes=1)
input_image = torch.randn(1, 3, 256, 256)  # Example input image
output = model(input_image)
print(output.shape)  # Check output shape
