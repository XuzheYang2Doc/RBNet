# Copyright (c) OpenMMLab. All rights reserved.
# Migrated from mmseg0.x: my_configs/unet.py

model = {'type': 'EncoderDecoderFull',
 'decode_head': {'type': 'UnetOrig',
                 'in_channels': 3,
                 'num_classes': 2,
                 'loss_decode': [{'type': 'CrossEntropyLoss',
                                  'loss_name': 'loss_ce',
                                  'use_sigmoid': False,
                                  'loss_weight': 2.0},
                                 {'type': 'DiceLoss',
                                  'loss_name': 'loss_dice',
                                  'loss_weight': 2.0}]},
 'train_cfg': {},
 'test_cfg': {'mode': 'slide', 'crop_size': (512, 512), 'stride': (256, 256)}}
