_base_ = [
    '../_base_/models/vit-base-p32.py',
    'huangfan2_bs64_pil_resize_384.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# model setting
model = dict(backbone=dict(img_size=384), head=dict(num_classes=3))

# dataset setting
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

# schedule setting
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(lr=0.003 / 10))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4 / 10,
        by_epoch=True,
        begin=0,
        end=30 // 5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=270 // 5,
        by_epoch=True,
        begin=30 // 5,
        end=300 // 5,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300 // 5, val_interval=1)

# custom
default_hooks = dict(checkpoint=dict(interval=5))
visualizer = dict(
    type="UniversalVisualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="WandbVisBackend")],
)
# load_from = "checkpoints/vit-base-p32_384_20210916-1b45d2cf.pth"
load_from = "checkpoints/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth"
