# Copyright (c) OpenMMLab. All rights reserved.

# Reformatted code for consistent style
# NOTE: a plain U-Net has no ImageNet-pretrained release (the standard model is
# trained from scratch); its encoder is not a ResNet, so no init_cfg is set here.
# Compare it either from scratch, or swap in a ResNet-encoder U-Net to match the
# other models, which all use ImageNet-pretrained backbones.

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
