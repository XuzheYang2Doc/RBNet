import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCN8(BaseDecodeHead):
    def __init__(self, out_channel=64, **kwargs):
        super(FCN8, self).__init__(**kwargs)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=24),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, padding=0)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96),

            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=96),

            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            nn.MaxPool2d(kernel_size=2, padding=0)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.MaxPool2d(kernel_size=2, padding=0)
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channel),

            nn.MaxPool2d(kernel_size=2, padding=0)
        )

        # k倍上采样
        self.upsample_2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, padding=1, stride=2)
        self.upsample_4 = nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=4,
                                             padding=0, stride=4)
        self.upsample_81 = nn.ConvTranspose2d(in_channels=128 + out_channel + 64, out_channels=128 + out_channel + 64,
                                              kernel_size=4, padding=0, stride=4)
        self.upsample_82 = nn.ConvTranspose2d(in_channels=128 + out_channel + 64, out_channels=128 + out_channel + 64,
                                              kernel_size=4, padding=1, stride=2)
        # 最后的预测模块
        self.final = nn.Sequential(
            nn.Conv2d(128 + out_channel + 64, out_channel, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = x.float()
        # conv1->pool1->输出
        x = self.stage1(x)
        # conv2->pool2->输出
        x = self.stage2(x)
        # conv3->pool3->输出输出, 经过上采样后, 需要用pool3暂存
        x = self.stage3(x)
        pool3 = x
        # conv4->pool4->输出输出, 经过上采样后, 需要用pool4暂存
        x = self.stage4(x)
        pool4 = self.upsample_2(x)

        x = self.stage5(x)
        conv7 = self.upsample_4(x)

        # 对所有上采样过的特征图进行concat, 在channel维度上进行叠加
        x = torch.cat([pool3, pool4, conv7], dim=1)
        # 经过一个分类网络,输出结果(这里采样到原图大小,分别一次2倍一次4倍上采样来实现8倍上采样)
        output = self.upsample_81(x)
        output = self.upsample_82(output)
        output = self.final(output)
        output = self.cls_seg(output)
        return output
