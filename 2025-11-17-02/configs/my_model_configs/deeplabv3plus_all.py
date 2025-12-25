# Copyright (c) OpenMMLab. All rights reserved.
# MMSegmentation 1.x config (migrated from mmseg0.x: my_configs/deeplabv3plus_all.py)

_base_ = [
    '../_base_/models/deeplabv3plus_all.py',
    '../_base_/datasets/rice_leaf_lesion_512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10k_poly_adam.py',
]

# Keep the same crop size as old config (used by slide inference)
crop_size = (512, 512)

# Move normalization/padding into the data preprocessor (recommended in MMSeg 1.x).
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.0, 0.0, 0.0],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(data_preprocessor=data_preprocessor)

# Keep this from old configs; safe even if you run single-GPU.
model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True)
