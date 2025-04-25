# models/srgan.py

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class SRGAN_Generator(nn.Module):
    def __init__(self, in_channels=3, num_res_blocks=16, upscale_factor=4):
        super(SRGAN_Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, 1, 4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        self.res_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

        up_blocks = []
        for _ in range(int(upscale_factor / 2)):
            up_blocks.append(UpsampleBlock(64, 2))
        self.upsample = nn.Sequential(*up_blocks)

        self.output = nn.Sequential(
            nn.Conv2d(64, in_channels, 9, 1, 4),
            nn.Tanh()
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.res_blocks(initial)
        x = self.res_conv(x)
        x = x + initial
        x = self.upsample(x)
        return self.output(x)

class SRGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SRGAN_Discriminator, self).__init__()

        def block(in_feat, out_feat, stride):
            return [
                nn.Conv2d(in_feat, out_feat, 3, stride, 1),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers = []
        layers += block(in_channels, 64, 1)
        layers += block(64, 64, 2)
        layers += block(64, 128, 1)
        layers += block(128, 128, 2)
        layers += block(128, 256, 1)
        layers += block(256, 256, 2)
        layers += block(256, 512, 1)
        layers += block(512, 512, 2)

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
