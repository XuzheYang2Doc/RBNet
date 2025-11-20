# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F
from collections import OrderedDict


@HEADS.register_module()
class CARUnet(BaseDecodeHead):
    def __init__(self, in_channel=3, num_classes=2, ca=False, denseaspp=False, densecadrb=False, **kwargs):
        super(CARUnet, self).__init__(**kwargs)
        self.ca = ca
        self.densecadrb = densecadrb
        # encoder
        if densecadrb:
            self.cadrb_encoder1 = DenseCADRB(in_channel, 16, ca=self.ca)
            self.cadrb_encoder2 = DenseCADRB(16, 32, ca=self.ca)
            self.cadrb_encoder3 = DenseCADRB(32, 64, ca=self.ca)
            self.cadrb_encoder4 = DenseCADRB(64, 64, ca=self.ca)
        else:
            self.cadrb_encoder1 = CADRB(in_channel, 16, ca=self.ca)
            self.cadrb_encoder2 = CADRB(16, 32, ca=self.ca)
            self.cadrb_encoder3 = CADRB(32, 64, ca=self.ca)
            self.cadrb_encoder4 = CADRB(64, 64, ca=self.ca)
        self.denseaspp = denseaspp
        if self.denseaspp:
            self.denseaspp_block = DenseASPPBlock(64, 256, 64)
        if self.ca:
            self.attention_blcok1_1 = SKAttention(channel=16)
            self.attention_blcok1_2 = SKAttention(channel=16)
            self.attention_blcok1_3 = SKAttention(channel=16)
            self.attention_blcok2_1 = SKAttention(channel=32)
            self.attention_blcok2_2 = SKAttention(channel=32)
            self.attention_blcok2_3 = SKAttention(channel=32)
            self.attention_blcok3_1 = SKAttention(channel=64)
            self.attention_blcok3_2 = SKAttention(channel=64)
            self.attention_blcok3_3 = SKAttention(channel=64)
        else:
            self.attention_blcok1 = MecaBlock(num_feature=16, ratio=4)
            self.attention_blcok2 = MecaBlock(num_feature=32, ratio=4)
            self.attention_blcok3 = MecaBlock(num_feature=64, ratio=4)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        # decoder
        self.cadrb_decoder3 = Up(128, 32, ca=self.ca, densecadrb=self.densecadrb)
        self.cadrb_decoder2 = Up(64, 16, ca=self.ca, densecadrb=self.densecadrb)
        self.cadrb_decoder1 = Up(32, 16, ca=self.ca, densecadrb=self.densecadrb)
        self.conv_seg = nn.Conv2d(16, num_classes, kernel_size=1)


    def forward(self, inputs):
        # encoder
        encoder_out1 = self.cadrb_encoder1(inputs)  # 512
        encoder_out1_down = self.max_pool(encoder_out1)  # 256
        encoder_out2 = self.cadrb_encoder2(encoder_out1_down)  # 256
        encoder_out2_down = self.max_pool(encoder_out2)  # 128
        encoder_out3 = self.cadrb_encoder3(encoder_out2_down)  # 128
        encoder_out4_down = self.max_pool(encoder_out3)  # 64
        encoder_out4 = self.cadrb_encoder4(encoder_out4_down)  # 64
        if self.denseaspp:
            encoder_out4 = self.denseaspp_block(encoder_out4)
        # docoder
        decoder_out3 = self.cadrb_decoder3(encoder_out4, encoder_out3)  # 128
        decoder_out2 = self.cadrb_decoder2(decoder_out3, encoder_out2)  # 256
        decoder_out1 = self.cadrb_decoder1(decoder_out2, encoder_out1)  # 512
        return self.conv_seg(decoder_out1)



class Up(nn.Module):
    def __init__(self, in_ch, out_ch, ca=False, densecadrb=False, bilinear=True):
        # 定义了self.up的方法
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)  # // 除以的结果向下取整
        if densecadrb:
            self.conv = DenseCADRB(in_ch, out_ch, ca=ca)
        else:
            self.conv = CADRB(in_ch, out_ch, ca=ca)

    def forward(self, x1, x2):  # x2是左侧的输出，x1是上一大层来的输出
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # 将两个tensor拼接在一起 dim=1：在通道数（C）上进行拼接
        x = self.conv(x)
        return x

class CADRB(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, activate=True, ca=False, **kwargs):
        super(CADRB, self).__init__(**kwargs)
        self.conv1_1 = ConvolutionBlockDropblock(in_channel, out_channel)
        self.conv1_2 = ConvolutionBlockDropblock(out_channel, out_channel)
        if ca:
            self.meca = CoordAtt(inp=out_channel, oup=out_channel)
        else:
            self.meca = MecaBlock(num_feature=out_channel, ratio=4)
        self.block_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv_final = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1, padding=0)
        self.activate = activate
        self.bn = nn.SyncBatchNorm(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        out = self.conv1_1(inputs)
        out = self.conv1_2(out)
        meca_score = self.meca(out)
        out = meca_score * out
        block_conv = self.block_conv(inputs)
        out = torch.concat([out, block_conv], dim=1)
        out = self.conv_final(out)
        if self.activate:
            out = self.bn(out)
            out = self.relu(out)
        return out


class DenseCADRB(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, activate=True, ca=False, **kwargs):
        super(DenseCADRB, self).__init__(**kwargs)
        self.conv1_1 = ConvolutionBlockDropblock(in_channel, out_channel)
        self.conv1_2 = ConvolutionBlockDropblock(out_channel, out_channel)
        if ca:
            self.meca1 = CoordAtt(inp=out_channel, oup=out_channel)
            self.meca2 = CoordAtt(inp=out_channel, oup=out_channel)
        else:
            self.meca1 = MecaBlock(num_feature=out_channel, ratio=4)
            self.meca2 = MecaBlock(num_feature=out_channel, ratio=4)
        self.block_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv_final = nn.Conv2d(out_channel * 3, out_channel, kernel_size=1, padding=0)
        self.activate = activate
        self.bn = nn.SyncBatchNorm(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        out1 = self.conv1_1(inputs)
        meca_score1 = self.meca1(out1)
        out1 = meca_score1 * out1
        out2 = self.conv1_2(out1)
        meca_score2 = self.meca2(out2)
        out2 = meca_score2 * out2
        block_conv = self.block_conv(inputs)
        out = torch.concat([out1, out2, block_conv], dim=1)
        out = self.conv_final(out)
        if self.activate:
            out = self.bn(out)
            out = self.relu(out)
        return out


class ResidualDropBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, activate=False, **kwargs):
        super(ResidualDropBlock, self).__init__(**kwargs)
        self.conv1_1 = ConvolutionBlockDropblock(in_channel, out_channel)
        self.conv1_2 = ConvolutionBlockDropblock(out_channel, out_channel, activate=False)
        self.block_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
        self.activate = activate
        self.bn = nn.SyncBatchNorm(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        out = self.conv1_1(inputs)
        block_conv = self.block_conv(inputs)
        out = out + block_conv
        res2 = out
        out = self.conv1_2(out)
        out = out + res2
        if self.activate:
            out = self.bn(out)
            out = self.relu(out)
        return out


class ConvolutionBlockDropblock(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, activate=True,  **kwargs):
        super(ConvolutionBlockDropblock, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn = nn.SyncBatchNorm(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.activate = activate

    def forward(self, inputs):
        out = self.bn(inputs)
        if self.activate:
            out = self.relu(out)
        out = self.conv1(out)
        return out


class MecaBlock(nn.Module):
    def __init__(self, num_feature, ratio=4):
        super(MecaBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_conv = nn.Linear(num_feature, num_feature, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(num_feature, num_feature // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_feature // ratio, num_feature, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        b, c, _, _ = input.size()
        y_avg_out = self.avg_pool(input).view(b, c)
        y_max_out = self.max_pool(input).view(b, c)
        avg_share_out = self.shared_conv(y_avg_out)
        max_share_out = self.shared_conv(y_max_out)
        out = avg_share_out + max_share_out
        score = self.fc(out).view(b, c, 1, 1)
        return score



# tensor = torch.randn((4, 3, 512, 512))
# conv = CARUnet()
# out = conv(tensor)
# print(out.shape)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.SyncBatchNorm(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return a_w * a_h


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.SyncBatchNorm, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.SyncBatchNorm, norm_kwargs=None):
        super(DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels + 5 * 64, inter_channels2, 1)
        )

    def forward(self, x):
        # x-64
        # aspp3-64
        aspp3 = self.aspp_3(x)
        # x-128
        x = torch.cat([aspp3, x], dim=1)
        #
        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        out = self.block(x)

        return out


class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=4, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.SyncBatchNorm(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = in_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, self.out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding https://github.com/iscyy/yoloair
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


def positional_encoding(seq_len, embed_dim):
    """
    创建位置编码矩阵
    Args:
        seq_len (int): 序列长度
        embed_dim (int): 嵌入维度
    Returns:
        torch.Tensor: 位置编码矩阵，形状为 (seq_len, embed_dim)
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
    position_encoding = torch.zeros(seq_len, embed_dim)
    position_encoding[:, 0::2] = torch.sin(position * div_term)
    position_encoding[:, 1::2] = torch.cos(position * div_term)
    return position_encoding
