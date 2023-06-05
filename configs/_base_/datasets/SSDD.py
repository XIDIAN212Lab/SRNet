# dataset settings
dataset_type = 'SSDD'
data_root = './data/SSDD/'
angle_version = 'le90'

img_norm_cfg = dict(
    mean=[21.55, 21.55, 21.55], std=[24.42, 24.42, 24.42], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RLoadAnnotations', with_mask=True),
    dict(type='RResize', img_scale=(608, 608)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='RRandomRotate',
        rotate_ratio=0.5,
        angles_range=30,
        auto_bound=False,
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', pad_to_square=True),
    dict(type='RDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', pad_to_square=True),
            dict(type='RDefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(type=dataset_type,
                     ann_file=data_root + 'annotations_coco/train.json',
                     img_prefix=data_root + 'images/train/',
                     pipeline=train_pipeline,
                     version=angle_version)
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_coco/test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_coco/test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline,
        version=angle_version))
