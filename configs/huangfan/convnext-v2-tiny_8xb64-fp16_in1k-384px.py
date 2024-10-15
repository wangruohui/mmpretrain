_base_ = [
    '../_base_/models/convnext_v2/tiny.py',
    'huangfan_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# model setting
model = dict(head=dict(num_classes=3))

# dataset setting
# train_dataloader = dict(batch_size=32)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=3.2e-3),
    clip_grad=None,
)
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=512.)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=40,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=40)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

visualizer = dict(
    type="UniversalVisualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="WandbVisBackend")],
)

# configure default hooks
default_hooks = dict(checkpoint=dict(interval=5))

load_from = "checkpoints/convnext-v2-tiny_3rdparty-fcmae_in1k_20230104-80513adc.pth"