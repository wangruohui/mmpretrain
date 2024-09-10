# dataset settings
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=5,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=1.0,
        dataset = dict(
            type='CustomDataset',
            data_root='data/',
            classes=['drawings', 'hentai', 'neutral', "porn", "sexy"],
            ann_file="nsfw_train_1.txt",
            pipeline=train_pipeline),
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='data/',
        classes=['drawings', 'hentai', 'neutral', "porn", "sexy"],
        ann_file="nsfw_val_1.txt",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
