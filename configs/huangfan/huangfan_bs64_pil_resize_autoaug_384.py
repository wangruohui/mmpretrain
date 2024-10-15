# dataset settings
data_preprocessor = dict(
    num_classes=3,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', ignore_empty=True),
    dict(
        type='RandomResizedCrop',
        crop_ratio_range=(0.2, 1.0),
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', ignore_empty=True),
    dict(
        type='ResizeEdge',
        scale=384,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='data/',
        classes=['normal', 'politics', 'porn'],
        ann_file="huangfan-shumei-train.txt",
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='data/',
        classes=['normal', 'politics', 'porn'],
        ann_file="huangfan-shumei-val.txt",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
