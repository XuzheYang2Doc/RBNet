# Modified from
# https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
#
# This work is licensed under the NVIDIA Source Code License.
#
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)
#
#  1. Definitions
#  "Licensor" means any person or entity that distributes its Work.
#  "Software" means the original work of authorship made available under
# this License.
#  "Work" means the Software and any additions to or derivative works of
# the Software that are made available under this License.
#  The terms "reproduce," "reproduction," "derivative works," and
# "distribution" have the meaning as provided under U.S. copyright law;
# provided, however, that for the purposes of this License, derivative
# works shall not include works that remain separable from, or merely
# link (or bind by name) to the interfaces of, the Work.
#  Works, including the Software, are "made available" under this License
# by including in or with the Work either (a) a copyright notice
# referencing the applicability of this License to the Work, or (b) a
# copy of this License.
#  2. License Grants
#      2.1 Copyright Grant. Subject to the terms and conditions of this
#     License, each Licensor grants to you a perpetual, worldwide,
#     non-exclusive, royalty-free, copyright license to reproduce,
#     prepare derivative works of, publicly display, publicly perform,
#     sublicense and distribute its Work and any resulting derivative
#     works in any form.
#  3. Limitations
#      3.1 Redistribution. You may reproduce or distribute the Work only
#     if (a) you do so under this License, (b) you include a complete
#     copy of this License with your distribution, and (c) you retain
#     without modification any copyright, patent, trademark, or
#     attribution notices that are present in the Work.
#      3.2 Derivative Works. You may specify that additional or different
#     terms apply to the use, reproduction, and distribution of your
#     derivative works of the Work ("Your Terms") only if (a) Your Terms
#     provide that the use limitation in Section 3.3 applies to your
#     derivative works, and (b) you identify the specific derivative
#     works that are subject to Your Terms. Notwithstanding Your Terms,
#     this License (including the redistribution requirements in Section
#     3.1) will continue to apply to the Work itself.
#      3.3 Use Limitation. The Work and any derivative works thereof only
#     may be used or intended for use non-commercially. Notwithstanding
#     the foregoing, NVIDIA and its affiliates may use the Work and any
#     derivative works commercially. As used herein, "non-commercially"
#     means for research or evaluation purposes only.
#      3.4 Patent Claims. If you bring or threaten to bring a patent claim
#     against any Licensor (including any claim, cross-claim or
#     counterclaim in a lawsuit) to enforce any patents that you allege
#     are infringed by any Work, then your rights under this License from
#     such Licensor (including the grant in Section 2.1) will terminate
#     immediately.
#      3.5 Trademarks. This License does not grant any rights to use any
#     Licensor’s or its affiliates’ names, logos, or trademarks, except
#     as necessary to reproduce the notices described in this License.
#      3.6 Termination. If you violate any term of this License, then your
#     rights under this License (including the grant in Section 2.1) will
#     terminate immediately.
#  4. Disclaimer of Warranty.
#  THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
# NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
# THIS LICENSE.
#  5. Limitation of Liability.
#  EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
# THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
# SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
# OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
# (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
# LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
# COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGES.

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', sae=False, tr=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.sae = sae
        if self.sae:
            self.sae_model = SaELayer(1024)
        self.tr = tr
        if self.tr:
            self.tr_model1 = CCTB(32, 64, 4, 4)
            self.tr_model2 = CCTB(64, 32, 4, 4)
            self.tr_model3 = CCTB(160, 16, 4, 4)
            self.tr_model4 = CCTB(256, 8, 4, 4)


    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        if self.tr:
            inputs[0] = self.tr_model1(inputs[0]) + inputs[0]
            inputs[1] = self.tr_model2(inputs[1]) + inputs[1]
            inputs[2] = self.tr_model3(inputs[2]) + inputs[2]
            inputs[3] = self.tr_model4(inputs[3]) + inputs[3]

        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        if self.sae:
            concat_feature = torch.cat(outs, dim=1)
            concat_feature = self.sae_model(concat_feature) + concat_feature
            out = self.fusion_conv(concat_feature)
        else:
            out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out


class SaELayer(nn.Module):
    def __init__(self, in_channel, reduction=2):
        super(SaELayer, self).__init__()
        assert in_channel >= reduction and in_channel % reduction == 0, 'invalid in_channel in SaElayer'
        self.reduction = reduction
        self.cardinality = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # cardinality 1
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 2
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 3
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 4
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channel // self.reduction * self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1, y2, y3, y4], dim=1)
        y_ex_dim = self.fc(y_concate).view(b, c, 1, 1)

        return x * y_ex_dim.expand_as(x)


import copy
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange
import numpy as np


class FCM(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FCM, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.BN = nn.BatchNorm2d(ch_out, eps=1e-5)
        self.Relu = nn.GELU()  # inplace=True

    def forward(self, x):
        x = self.conv(x)
        x = self.Relu(self.BN(x))
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out, eps=1e-5),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out, eps=1e-5),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out, eps=1e-5),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CCTB(nn.Module):
    def __init__(self, dim=256, reso=8, num_heads=8,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.LEM = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(dim, eps=1e-5),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        )
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        cnn_feat = self.LEM(x) + x
        H = W = self.patches_resolution
        feat = rearrange(cnn_feat, 'b c h w-> b (h w) c ', h=H, w=W)
        x = rearrange(x, 'b c h w-> b (h w) c ', h=H, w=W)
        B, L, C = feat.shape

        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(feat)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, (3, 3), (2, 2), (1, 1))
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        x = x.transpose(-2, -1).contiguous().view(B, C, H // 2, W // 2)

        return x


class AGBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AGBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=3,
                      stride=1,
                      padding=3 // 2,
                      bias=False),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=5,
                      stride=1,
                      padding=5 // 2,
                      bias=False),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi + x
