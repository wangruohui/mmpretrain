# dataset settings
data_preprocessor = dict(
    num_classes=5,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

imsize=336

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=336,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=6,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=336,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=336),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_root='data/',
        classes=['drawings', 'hentai', 'neutral', "porn", "sexy"],
        ann_file="nsfw_train_1.txt",
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_root='data/',
        classes=['drawings', 'hentai', 'neutral', "porn", "sexy"],
        ann_file="nsfw_val_1.txt",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
# val_evaluator = dict(type='Accuracy', topk=(1, 5))
val_evaluator = [dict(type="Accuracy", topk=(1, )), dict(type="SingleLabelMetric")]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
