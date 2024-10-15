# dataset settings
data_preprocessor = dict(
    num_classes=3,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

imscale = 384

train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', ignore_empty=True),
    dict(type='RandomResizedCrop', scale=imscale, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', ignore_empty=True),
    dict(type='ResizeEdge', scale=imscale, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=imscale),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='data/',
        classes=['normal', 'politics', 'porn'],
        ann_file="huangfan-shumei-0929-train.txt",
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
        ann_file="huangfan-shumei-0929-val.txt",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='/mnt/storage/user/wangruohui/涉政与色情低俗图包',
        classes=['normal', 'politics', 'porn'],
        ann_file="/mnt/storage/user/wangruohui/涉政与色情低俗图包/ann_dana.txt",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator
