# Copyright (c) OpenMMLab. All rights reserved.
# Migrated from mmseg0.x configs in my_model_configs.tar

# Dataset settings (custom dataset)
dataset_type = 'MyDataset'
data_root = 'data/semantic/'  # keep relative to your project root
crop_size = (512, 512)
img_size = (1024, 1024)

# Data pipeline
# NOTE:
# - In MMSeg 1.x, normalization/padding are recommended to be handled by model.data_preprocessor.
# - Resize args: img_scale -> scale; training uses RandomResize.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=crop_size, ratio_range=(1.0, 1.0), keep_ratio=False),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Dataloaders
# IterBasedTrainLoop + InfiniteSampler is the common default in MMSeg 1.x.
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='train/images1024',
                seg_map_path='train/labels1024'),
            pipeline=train_pipeline,
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images1024',
            seg_map_path='val/labels1024'),
        pipeline=test_pipeline,
        test_mode=True,
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images1024',
            seg_map_path='test/labels1024'),
        pipeline=test_pipeline,
        test_mode=True,
    )
)

# Evaluators (match old metrics: mIoU + mDice + mFscore)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
