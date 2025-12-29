# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UnetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, channel_list=None):
        super().__init__()
        if channel_list is None:
            channel_list = [32, 64, 128, 256, 512]
        self.inc = InConv(in_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1])
        self.down2 = Down(channel_list[1], channel_list[2])
        self.down3 = Down(channel_list[2], channel_list[3])
        self.down4 = Down(channel_list[3], channel_list[4])

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


class UnetHead(nn.Module):
    def __init__(self, channels: int, decoder_channel=None):
        super().__init__()
        if decoder_channel is None:
            decoder_channel = [512, 256, 128, 64, 32]

        self.up1 = Up(decoder_channel[0] + decoder_channel[1], decoder_channel[1])
        self.up2 = Up(decoder_channel[1] + decoder_channel[2], decoder_channel[2])
        self.up3 = Up(decoder_channel[2] + decoder_channel[3], decoder_channel[3])
        self.up4 = Up(decoder_channel[3] + decoder_channel[4], decoder_channel[4])
        self.head = nn.Conv2d(decoder_channel[4], channels, kernel_size=3, padding=1)

    def forward(self, inputs) -> torch.Tensor:
        out = self.up1(inputs[4], inputs[3])
        out = self.up2(out, inputs[2])
        out = self.up3(out, inputs[1])
        out = self.up4(out, inputs[0])
        return self.head(out)


@MODELS.register_module()
class UnetOrig(BaseDecodeHead):
    """Migrated from mmseg0.x.

    This head consumes the raw image tensor (N, C, H, W).
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 biformer: bool = False,
                 ks: bool = False,
                 **kwargs):
        super().__init__(in_channels=in_channels, channels=channels, **kwargs)
        self.biformer = biformer
        self.ks = ks
        if self.biformer or self.ks:
            raise NotImplementedError(
                'This repo migration only supports biformer=False and ks=False.')

        self.backbone = UnetBackbone(in_channels=in_channels)
        self.head = UnetHead(channels=channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_list = self.backbone(inputs)
        out = self.head(feature_list)
        return self.cls_seg(out)
