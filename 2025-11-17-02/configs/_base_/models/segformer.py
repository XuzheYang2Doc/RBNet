# Copyright (c) OpenMMLab. All rights reserved.
# Migrated from mmseg0.x: my_configs/segformer.py

model = {
    'type': 'EncoderDecoder',
    'backbone': {
        'type': 'MixVisionTransformer',
        'in_channels': 3,
        'embed_dims': 32,
        'num_stages': 4,
        'num_layers': [2, 2, 2, 2],
        'num_heads': [1, 2, 5, 8],
        'patch_sizes': [7, 3, 3, 3],
        'sr_ratios': [8, 4, 2, 1],
        'out_indices': (0, 1, 2, 3),
        'mlp_ratio': 4,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1
    },
    'decode_head': {
        'type': 'SegformerHead',
        'in_channels': [32, 64, 160, 256],
        'in_index': [0, 1, 2, 3],
        'channels': 256,
        'dropout_ratio': 0.1,
        'num_classes': 2,
        'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
        'align_corners': False,
        'loss_decode': [
            {
                'type': 'CrossEntropyLoss',
                'loss_name': 'loss_ce',
                'use_sigmoid': False,
                'loss_weight': 2.0
            },
            {
                'type': 'DiceLoss',
                'loss_name': 'loss_dice',
                'loss_weight': 2.0
            }
        ]
    },
    'train_cfg': {},
    'test_cfg': {'mode': 'slide', 'crop_size': (512, 512), 'stride': (256, 256)}
}
