# Copyright (c) OpenMMLab. All rights reserved.

# Reformatted code for consistent style

model = dict(
    type='EncoderDecoderFull',
    decode_head=dict(
        type='UnetOrig',
        in_channels=3,
        num_classes=2,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=2.0),
            dict(
                type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide', crop_size=(512, 512), stride=(256, 256)))
