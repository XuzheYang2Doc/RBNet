# Copyright (c) OpenMMLab. All rights reserved.
# Migrated from mmseg0.x: my_configs/deeplabv3plus_tr.py

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet18_v1c')),
    decode_head=dict(
        type='DepthwiseSeparableASPPHeadSAETR',
        tr=True,
        in_channels=512,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=64,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=2.0),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=2.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))
