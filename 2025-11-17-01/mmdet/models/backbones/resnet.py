# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer, ContextBlock
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..layers import ResLayer


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@MODELS.register_module()
class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 pe=False,
                 context=False):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.pe = pe
        self.context = context
        if self.pe:
            self.pe_model = PENet()
        if self.context:
            self.context_model1 = ConvSwinMerge(256, 256)
            self.context_model2 = ConvSwinMerge(512, 512)
        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.pe:
            x = self.pe_model(x) + x
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.context:
            outs[0] = self.context_model1(outs[0])
            outs[1] = self.context_model2(outs[1])
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)


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
    def __init__(self, inp, oup, reduction=2):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.bn1 = SyncBatchNorm(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        # y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return a_w * a_h


class ConvSwinMerge(nn.Module):
    def __init__(self, inp, oup):
        super(ConvSwinMerge, self).__init__()
        self.coord_att = CoordAtt(inp, oup)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.sae = SaELayer(oup)

    def forward(self, x):
        out = self.coord_att(x) + x
        out = self.conv(out) + out
        out = self.sae(out)
        return out

from functools import lru_cache
from einops import einsum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
from typing import *
from torch.autograd import Function



class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv_x = nn.Conv2d(in_features, out_features, 3, padding=1)

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return self.conv_x(x + self.block(x))


class PENet(nn.Module):
    def __init__(self,
                 num_high=3,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)

        for i in range(0, self.num_high + 1):
            self.__setattr__('AE_{}'.format(i), KANConv2d(3, 3, kernel_size=3, padding=1))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []

        for i in range(self.num_high + 1):
            trans_pyr = self.__getattr__('AE_{}'.format(i))(
                pyrs[-1 - i])
            trans_pyrs.append(trans_pyr)
        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        return out


class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator, train_grid, train_inv_denominator):
        # Compute the forward pass
        # print('\n')
        # print(f"Forward pass - grid: {(grid[0].item(),grid[-1].item())}, inv_denominator: {inv_denominator.item()}")

        # print(f"grid.shape: {grid.shape }")
        # print(f"grid: {(grid[0],grid[-1]) }")
        # print(f"inv_denominator.shape: {inv_denominator.shape }")
        # print(f"inv_denominator: {inv_denominator }")
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator)
        tanh_diff = torch.tanh(diff)
        tanh_diff_deriviative = -tanh_diff.mul(tanh_diff) + 1  # sech^2(x) = 1 - tanh^2(x)

        # Save tensors for backward pass
        ctx.save_for_backward(input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator)
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator

        return tanh_diff_deriviative

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator = ctx.saved_tensors
        grad_grid = None
        grad_inv_denominator = None

        # print(f"tanh_diff_deriviative shape: {tanh_diff_deriviative.shape }")
        # print(f"tanh_diff shape: {tanh_diff.shape }")
        # print(f"grad_output shape: {grad_output.shape }")

        # Compute the backward pass for the input
        grad_input = -2 * tanh_diff * tanh_diff_deriviative * grad_output
        # print(f"Backward pass 1 - grad_input: {(grad_input.min().item(), grad_input.max().item())}")
        # print(f"grad_input shape: {grad_input.shape }")
        # print(f"grad_input.sum(dim=-1): {grad_input.sum(dim=-1).shape}")
        grad_input = grad_input.sum(dim=-1).mul(inv_denominator)
        # print(f"Backward pass 2 - grad_input: {(grad_input.min().item(), grad_input.max().item())}")
        # print(f"grad_input: {grad_input}")
        # print(f"grad_input shape: {grad_input.shape }")

        # Compute the backward pass for grid
        if ctx.train_grid:
            # print('\n')
            # print(f"grad_grid shape: {grad_grid.shape }")
            grad_grid = -inv_denominator * grad_output.sum(dim=0).sum(
                dim=0)  # -(inv_denominator * grad_output * tanh_diff_deriviative).sum(dim=0) #-inv_denominator * grad_output.sum(dim=0).sum(dim=0)
            # print(f"Backward pass - grad_grid: {(grad_grid[0].item(),grad_grid[-1].item())}")
            # print(f"grad_grid.shape: {grad_grid.shape }")
            # print(f"grad_grid: {(grad_grid[0],grad_grid[-1]) }")
            # print(f"inv_denominator shape: {inv_denominator.shape }")
            # print(f"grad_grid shape: {grad_grid.shape }")

        # Compute the backward pass for inv_denominator
        if ctx.train_inv_denominator:
            grad_inv_denominator = (
                    grad_output * diff).sum()  # (grad_output * diff * tanh_diff_deriviative).sum() #(grad_output* diff).sum()
            # print(f"Backward pass - grad_inv_denominator: {grad_inv_denominator.item()}")
            # print(f"diff shape: {diff.shape }")

            # print(f"grad_inv_denominator shape: {grad_inv_denominator.shape }")
            # print(f"grad_inv_denominator : {grad_inv_denominator }")

        return grad_input, grad_grid, grad_inv_denominator, None, None  # same number as tensors or parameters


class ReflectionalSwitchFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -1.2,
            grid_max: float = 0.2,
            num_grids: int = 8,
            exponent: int = 2,
            inv_denominator: float = 0.5,
            train_grid: bool = False,
            train_inv_denominator: bool = False,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.train_grid = torch.tensor(train_grid, dtype=torch.bool)
        self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool)
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        # print(f"grid initial shape: {self.grid.shape }")
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32),
                                                  requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def forward(self, x):
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator, self.train_grid, self.train_inv_denominator)


class KANLayer(nn.Module):
    def __init__(self, input_features, output_features, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # The number of points in the grid for the spline interpolation.
        self.grid_size = grid_size
        # The order of the spline used in the interpolation.
        self.spline_order = spline_order
        # Activation function used for the initial transformation of the input.
        self.base_activation = base_activation()
        # The range of values over which the grid for spline interpolation is defined.
        self.grid_range = grid_range

        # Initialize the base weights with random values for the linear transformation.
        self.base_weight = nn.Parameter(torch.randn(output_features, input_features))
        # Initialize the spline weights with random values for the spline transformation.
        self.spline_weight = nn.Parameter(torch.randn(output_features, input_features, grid_size + spline_order))
        # Add a layer normalization for stabilizing the output of this layer.
        self.layer_norm = nn.LayerNorm(output_features)
        # Add a PReLU activation for this layer to provide a learnable non-linearity.
        self.prelu = nn.PReLU()

        # Compute the grid values based on the specified range and grid size.
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        ).expand(input_features, -1).contiguous()

        # Initialize the weights using Kaiming uniform distribution for better initial values.
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity='linear')

    def forward(self, x):
        # Process each layer using the defined base weights, spline weights, norms, and activations.
        grid = self.grid.to(x.device)
        # Move the input tensor to the device where the weights are located.

        # Perform the base linear transformation followed by the activation function.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype).to(x.device)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[:, :-(k + 1)]
            right_intervals = grid[:, k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        bases = bases.contiguous()

        # Compute the spline transformation and combine it with the base transformation.
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weight.view(self.spline_weight.size(0), -1))
        # Apply layer normalization and PReLU activation to the combined output.
        x = self.prelu(self.layer_norm(base_output + spline_output))

        return x


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=nn.SiLU,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation()
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # x = torch.tanh(x)
        x = torch.clamp(x, -1.0, 1.0)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class GRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degree=3, act=nn.SiLU):
        super(GRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degree

        self.act = act()

        self.norm = nn.LayerNorm(out_channels, dtype=torch.float32)

        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        self.grams_basis_weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, degree + 1, dtype=torch.float32)
        )

        self.base_weights = nn.Parameter(
            torch.zeros(out_channels, in_channels, dtype=torch.float32)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_channels * (self.degrees + 1.0)),
        )

        nn.init.xavier_uniform_(self.grams_basis_weights)

        nn.init.xavier_uniform_(self.base_weights)

    def beta(self, n, m):
        return (
                ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        return torch.stack(grams_basis, dim=-1)

    def forward(self, x):

        basis = F.linear(self.act(x), self.base_weights)

        x = torch.tanh(x).contiguous()

        grams_basis = self.act(self.gram_poly(x, self.degrees))

        y = einsum(
            grams_basis,
            self.grams_basis_weights,
            "b l d, l o d -> b o",
        )

        y = self.act(self.norm(y + basis))

        y = y.view(-1, self.out_channels)

        return y


class WavKANLayer(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WavKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features,
                                                 in_features))  # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real

        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window

            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")
        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        wavelet_output = wavelet_weighted.sum(dim=2)
        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        base_output = F.linear(self.base_activation(x), self.weight1)

        # base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output + base_output

        # Apply batch normalization
        return self.bn(combined_output)


class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0, act=nn.SiLU):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.act = act()
        self.norm = nn.LayerNorm(output_dim, dtype=torch.float32)

        self.base_weights = nn.Parameter(
            torch.zeros(output_dim, input_dim, dtype=torch.float32)
        )

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        nn.init.xavier_uniform_(self.base_weights)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)

        basis = F.linear(self.act(x), self.base_weights)

        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                    2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                    i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :,
                                                                                                  i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)

        y = self.act(self.norm(y + basis))
        return y


class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4 * g * g / ((k + 1) * (k + 1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k + 1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g + k, input_size))

    def forward(self, x):
        # Expand dimensions of x to match the shape of self.phase_low
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))

        # Perform the subtraction with broadcasting
        x1 = torch.relu(x_expanded - self.phase_low)
        x2 = torch.relu(self.phase_height - x_expanded)

        # Continue with the rest of the operations
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        # x = x.reshape((len(x), self.output_size, 1))
        x = x.reshape((len(x), self.output_size))
        return x


class SplineLinear_fstr(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)  # Using Xavier Uniform initialization


class FasterKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -1.2,
            grid_max: float = 0.2,
            num_grids: int = 8,
            exponent: int = 2,
            inv_denominator: float = 0.5,
            train_grid: bool = False,
            train_inv_denominator: bool = False,
            # use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.667,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = ReflectionalSwitchFunction(grid_min, grid_max, num_grids, exponent, inv_denominator, train_grid,
                                              train_inv_denominator)
        self.spline_linear = SplineLinear_fstr(input_dim * num_grids, output_dim, spline_weight_init_scale)
        # self.use_base_update = use_base_update
        # if use_base_update:
        #    self.base_activation = base_activation
        #    self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # print("Shape before LayerNorm:", x.shape)  # Debugging line to check the input shape
        x = self.layernorm(x)
        # print("Shape After LayerNorm:", x.shape)
        spline_basis = self.rbf(x).view(x.shape[0], -1)
        # print("spline_basis:", spline_basis.shape)

        # print("-------------------------")
        # ret = 0
        ret = self.spline_linear(spline_basis)
        # print("spline_basis.shape[:-2]:", spline_basis.shape[:-2])
        # print("*spline_basis.shape[:-2]:", *spline_basis.shape[:-2])
        # print("spline_basis.view(*spline_basis.shape[:-2], -1):", spline_basis.view(*spline_basis.shape[:-2], -1).shape)
        # print("ret:", ret.shape)
        # print("-------------------------")
        # if self.use_base_update:
        # base = self.base_linear(self.base_activation(x))
        # print("self.base_activation(x):", self.base_activation(x).shape)
        # print("base:", base.shape)
        # print("@@@@@@@@@")
        # ret += base
        return ret


class RBFLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_min=-2., grid_max=2., num_grids=8, spline_weight_init_scale=0.1):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=False)
        self.spline_weight = nn.Parameter(torch.randn(in_features * num_grids, out_features) * spline_weight_init_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        basis = torch.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
        return basis.reshape(basis.size(0), -1).matmul(self.spline_weight)


class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_min=-2., grid_max=2., num_grids=8, use_base_update=True,
                 base_activation=nn.SiLU(), spline_weight_init_scale=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.rbf_linear = RBFLinear(input_dim, output_dim, grid_min, grid_max, num_grids, spline_weight_init_scale)
        self.base_linear = nn.Linear(input_dim, output_dim) if use_base_update else None

    def forward(self, x):
        ret = self.rbf_linear(x)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(KANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = KANLinear(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out


class ChebyKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, degree=4):
        super(ChebyKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = ChebyKANLayer(in_channels * kernel_size * kernel_size, out_channels, degree=degree)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out


class FastKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FastKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = FastKANLayer(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out


class GRAMKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GRAMKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = GRAMLayer(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out


class WavKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, wavelet_type='mexican_hat'):
        super(WavKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = WavKANLayer(in_channels * kernel_size * kernel_size, out_channels, wavelet_type=wavelet_type)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out


class JacobiKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, degree=4):
        super(JacobiKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = JacobiKANLayer(in_channels * kernel_size * kernel_size, out_channels, degree=degree)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out


class ReLUKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ReLUKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = ReLUKANLayer(in_channels * kernel_size * kernel_size, 5, 3, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out


class FasterKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FasterKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = FasterKANLayer(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out


class RBFKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(RBFKANConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kanlayer = RBFKANLayer(in_channels * kernel_size * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        assert in_channels == self.in_channels

        # Apply unfold to get sliding local blocks
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)

        out_unfold = self.kanlayer(x_unfold)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))
        out = out_unfold.transpose(1, 2)
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_height, out_width)

        return out
