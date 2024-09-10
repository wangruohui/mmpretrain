# dataset settings
data_preprocessor = dict(
    num_classes=5,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

imsize=448

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=448, edge='short'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

test_pipeline_no_crop = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=448, edge='short'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type="CustomDataset",
        data_root="data/",
        classes=['drawings', 'hentai', 'neutral', "porn", "sexy"],
        ann_file="nsfw_train_1.txt",
        with_label=True,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type="CustomDataset",
        data_root="data/",
        classes=['drawings', 'hentai', 'neutral', "porn", "sexy"],
        ann_file="nsfw_val_1.txt",
        with_label=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=False,
    dataset=dict(
        type="CustomDataset",
        data_root="data/",
        classes=['drawings', 'hentai', 'neutral', "porn", "sexy"],
        ann_file="nsfw_val_1.txt",
        with_label=True,
        pipeline=test_pipeline_no_crop,
    ),
)

val_evaluator = [dict(type="Accuracy", topk=(1,)), dict(type="SingleLabelMetric")]

# If you want standard test, please manually configure the test dataset
test_evaluator = val_evaluator
