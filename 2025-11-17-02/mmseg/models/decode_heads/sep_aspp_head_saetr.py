# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .aspp_head import ASPPHead
from .sep_aspp_head import DepthwiseSeparableASPPModule


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class TopkRouting(nn.Module):
    """Differentiable topk routing with scaling."""

    def __init__(
        self,
        qk_dim: int,
        topk: int = 4,
        qk_scale: float | None = None,
        param_routing: bool = False,
        diff_routing: bool = False,
    ):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim**-0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)
        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight: str = 'none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: torch.Tensor, r_weight: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # r_idx: (n, p^2, topk)
        # r_weight: (n, p^2, topk)
        # kv: (n, p^2, w^2, c)
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv),
        )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim: int, qk_dim: int, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        n_win: int = 7,
        qk_dim: int | None = None,
        qk_scale: float | None = None,
        kv_per_win: int = 4,
        kv_downsample_ratio: int = 4,
        kv_downsample_kernel: int | None = None,
        kv_downsample_mode: str = 'identity',
        topk: int = 4,
        param_attention: str = 'qkvo',
        param_routing: bool = False,
        diff_routing: bool = False,
        soft_routing: bool = False,
        side_dwconv: int = 3,
        auto_pad: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0
        self.scale = qk_scale or self.qk_dim**-0.5

        self.lepe = (
            nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim)
            if side_dwconv > 0
            else lambda x: torch.zeros_like(x)
        )

        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        assert not (self.param_routing and not self.diff_routing)

        self.router = TopkRouting(
            qk_dim=self.qk_dim,
            qk_scale=self.scale,
            topk=self.topk,
            diff_routing=self.diff_routing,
            param_routing=self.param_routing,
        )
        if self.soft_routing:
            mul_weight = 'soft'
        elif self.diff_routing:
            mul_weight = 'hard'
        else:
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not supported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsample_mode} is not supported!')

        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad = auto_pad

    def forward(self, x: torch.Tensor, ret_attn_mask: bool = False):
        # x: NHWC
        if self.auto_pad:
            n, h_in, w_in, c = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - w_in % self.n_win) % self.n_win
            pad_b = (self.n_win - h_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, h, w, _ = x.size()
        else:
            n, h, w, c = x.size()
            assert h % self.n_win == 0 and w % self.n_win == 0

        j = i = self.n_win
        wh = h // j
        ww = w // i
        p2 = j * i

        # (n, (j*h), (i*w), c) -> (n, (j*i), h, w, c)
        x = x.view(n, j, wh, i, ww, c).permute(0, 1, 3, 2, 4, 5).contiguous().view(n, p2, wh, ww, c)

        q, kv = self.qkv(x)

        q_pix = q.reshape(n, p2, wh * ww, self.qk_dim)

        kv_nchw = kv.permute(0, 1, 4, 2, 3).reshape(n * p2, self.qk_dim + self.dim, wh, ww)
        kv_down = self.kv_down(kv_nchw)
        _, _, hk, wk = kv_down.shape
        kv_pix = kv_down.view(n, p2, self.qk_dim + self.dim, hk * wk).permute(0, 1, 3, 2).contiguous()

        q_win = q.mean([2, 3])
        k_win = kv[..., 0 : self.qk_dim].mean([2, 3])

        v_for_lepe = kv[..., self.qk_dim :]
        v_for_lepe = (
            v_for_lepe.view(n, j, i, wh, ww, self.dim)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(n, self.dim, j * wh, i * ww)
        )
        lepe = self.lepe(v_for_lepe).permute(0, 2, 3, 1).contiguous()

        r_weight, r_idx = self.router(q_win, k_win)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        m = self.num_heads
        qk_head_dim = self.qk_dim // m
        v_head_dim = self.dim // m
        topk = k_pix_sel.size(2)
        w2 = k_pix_sel.size(3)

        k_pix_sel = (
            k_pix_sel.view(n, p2, topk, w2, m, qk_head_dim)
            .permute(0, 1, 4, 5, 2, 3)
            .contiguous()
            .view(n * p2, m, qk_head_dim, topk * w2)
        )
        v_pix_sel = (
            v_pix_sel.view(n, p2, topk, w2, m, v_head_dim)
            .permute(0, 1, 4, 2, 3, 5)
            .contiguous()
            .view(n * p2, m, topk * w2, v_head_dim)
        )
        q_pix = (
            q_pix.view(n, p2, wh * ww, m, qk_head_dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(n * p2, m, wh * ww, qk_head_dim)
        )

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel

        out = (
            out.view(n, j, i, m, wh, ww, v_head_dim)
            .permute(0, 1, 4, 2, 5, 3, 6)
            .contiguous()
            .view(n, j * wh, i * ww, m * v_head_dim)
        )

        out = out + lepe
        out = self.wo(out)

        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :h_in, :w_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        return out


class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class BiformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = -1,
        num_heads: int = 8,
        n_win: int = 4,
        qk_dim: int | None = None,
        qk_scale: float | None = None,
        kv_per_win: int = 4,
        kv_downsample_ratio: int = 4,
        kv_downsample_kernel: int | None = None,
        kv_downsample_mode: str = 'ada_avgpool',
        topk: int = 4,
        param_attention: str = 'qkvo',
        param_routing: bool = False,
        diff_routing: bool = False,
        soft_routing: bool = False,
        mlp_ratio: int = 4,
        mlp_dwconv: bool = False,
        side_dwconv: int = 5,
        before_attn_dwconv: int = 3,
        pre_norm: bool = True,
        auto_pad: bool = False,
    ):
        super().__init__()
        qk_dim = qk_dim or dim
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = BiLevelRoutingAttention(
            dim=dim,
            num_heads=num_heads,
            n_win=n_win,
            qk_dim=qk_dim,
            qk_scale=qk_scale,
            kv_per_win=kv_per_win,
            kv_downsample_ratio=kv_downsample_ratio,
            kv_downsample_kernel=kv_downsample_kernel,
            kv_downsample_mode=kv_downsample_mode,
            topk=topk,
            param_attention=param_attention,
            param_routing=param_routing,
            diff_routing=diff_routing,
            soft_routing=soft_routing,
            side_dwconv=side_dwconv,
            auto_pad=auto_pad,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class SaELayer(nn.Module):
    def __init__(self, in_channel: int, reduction: int = 4):
        super().__init__()
        assert in_channel >= reduction and in_channel % reduction == 0
        self.reduction = reduction
        self.cardinality = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(nn.Linear(in_channel, in_channel // self.reduction, bias=False), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(in_channel, in_channel // self.reduction, bias=False), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(in_channel, in_channel // self.reduction, bias=False), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(in_channel, in_channel // self.reduction, bias=False), nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.Linear(in_channel // self.reduction * self.cardinality, in_channel, bias=False), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1, y2, y3, y4], dim=1)
        y_ex_dim = self.fc(y_concate).view(b, c, 1, 1)
        return x * y_ex_dim.expand_as(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp: int, oup: int, reduction: int = 4):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class CoordSaeLayer(nn.Module):
    def __init__(self, inp: int, oup: int):
        super().__init__()
        self.coord_att = CoordAtt(inp, oup)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.sae = SaELayer(oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.coord_att(x) + x
        out = self.conv(out) + out
        out = self.sae(out)
        return out


@MODELS.register_module()
class DepthwiseSeparableASPPHeadSAETR(ASPPHead):
    """DeepLabV3+ head with legacy SAE/TR branches migrated from mmseg0.x.

    - `tr=True`: apply BiFormer-style attention on ASPP concatenated features.
    - `sae=True`: apply CoordSaeLayer on features after concatenating low-level c1 features.
    """

    def __init__(
        self,
        c1_in_channels: int,
        c1_channels: int,
        sae: bool = False,
        tr: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert c1_in_channels >= 0
        self.tr = bool(tr)
        self.sae = bool(sae)

        aspp_concat_channels = (len(self.dilations) + 1) * self.channels
        post_cat_channels = self.channels + c1_channels

        if self.tr:
            self.tr_model = BiformerBlock(aspp_concat_channels)
        else:
            self.tr_model = None
        if self.sae:
            self.sae_model = CoordSaeLayer(post_cat_channels, post_cat_channels)
        else:
            self.sae_model = None

        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                post_cat_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        if self.tr_model is not None:
            aspp_outs = self.tr_model(aspp_outs) + aspp_outs

        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )
            output = torch.cat([output, c1_output], dim=1)
            if self.sae_model is not None:
                output = self.sae_model(output) + output

        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output
