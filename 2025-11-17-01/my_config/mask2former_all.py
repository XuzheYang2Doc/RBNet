# Mask2Former 单文件配置（精简版，保持可运行）
# 数据集参数
dataset_type = 'CocoDataset'
data_root = 'data/coco1/'
num_things_classes = 2
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (1024, 1024)

# 数据增强
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True,
        resize_type='Resize'),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        allow_negative_crop=True,
        recompute_bbox=True),
    dict(
        type='FilterAnnotations',
        by_mask=True,
        min_gt_bbox_wh=(1e-05, 1e-05)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape',
                   'img_shape', 'scale_factor'))
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train80.json',
        data_prefix=dict(img='train80/', seg='annotations/panoptic_train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/', seg='annotations/panoptic_val2017/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    classwise=True)

test_evaluator = val_evaluator

# 模型
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]

model = dict(
    type='Mask2Former',   # ⭐ 一定要保留
    backbone=dict(
        type='ResNet',
        context=True,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
        batch_augments=batch_augments),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],
        feat_channels=256,
        out_channels=256,
        num_queries=100,
        num_classes=num_classes,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            positional_encoding=dict(num_feats=128, normalize=True),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256, num_heads=8, num_levels=3, num_points=4,
                        batch_first=True, dropout=0.0),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=1024, num_fcs=2,
                        ffn_drop=0.0, act_cfg=dict(type='ReLU', inplace=True))))),
        tr=True,
        transformer_decoder=dict(
            num_layers=4,
            return_intermediate=True,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_heads=8, batch_first=True, dropout=0.0),
                cross_attn_cfg=dict(embed_dims=256, num_heads=8, batch_first=True, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, num_fcs=2,
                    ffn_drop=0.0, act_cfg=dict(type='ReLU', inplace=True)))),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=5.0,
            reduction='mean'),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
            reduction='mean')),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)]),
        sampler=dict(type='MaskPseudoSampler'),
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75),
    test_cfg=dict(
        instance_on=True,
        semantic_on=False,
        panoptic_on=False,
        filter_low_score=True,
        iou_thr=0.8,
        max_per_image=100))

# 训练参数
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        custom_keys=dict(
            backbone=dict(lr_mult=0.1, decay_mult=1.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0))))

param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    begin=0,
    end=25000,
    gamma=0.1,
    milestones=[20000])  # 学习率分别在 7k、9k iter 时衰减

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=25000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=3,
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

log_processor = dict(type='LogProcessor', by_epoch=False, window_size=50)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends)

# 运行参数
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
find_unused_parameters=True
log_level = 'INFO'
load_from = "/data/code2025/Q2/2025-11-17-01/work_dirs_temp/mask2former_all/iter_10000.pth"
resume = False
launcher = 'pytorch'
work_dir = f'./work_dirs/{{ fileBasenameNoExtension }}'
