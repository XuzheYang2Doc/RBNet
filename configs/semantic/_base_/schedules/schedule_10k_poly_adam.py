# Copyright (c) OpenMMLab. All rights reserved.
# Migrated from mmseg0.x: optimizer + poly lr + IterBasedRunner(max_iters=10000)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3)
)

# lr_config(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False) -> PolyLR
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-5, power=0.9, begin=0, end=10000, by_epoch=False)
]

# runner -> loops
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
