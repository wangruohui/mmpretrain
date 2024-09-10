_base_ = [
    '../_base_/models/vit-base-p16.py',
    # 'nsfw_bs64_pil_resize_336.py',
    'nsfw_bs64_rsb_a3_336.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]

# model setting
model = dict(backbone=dict(img_size=384))

# dataset setting
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', scale=384, backend='pillow'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='ResizeEdge', scale=384, edge='short', backend='pillow'),
#     dict(type='CenterCrop', crop_size=384),
#     dict(type='PackInputs'),
# ]

# train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))
# val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=30//2,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=270//2,
        by_epoch=True,
        begin=30//2,
        end=300//2,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300//2, val_interval=5)

# schedule setting
# optimizer=dict(lr=0.0003),
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

# visualizer
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')],
)

load_from = 'checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'