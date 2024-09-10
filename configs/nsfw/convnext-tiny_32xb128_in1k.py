_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    'nsfw_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# dataset setting
train_dataloader = dict(batch_size=128)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    # optimizer=dict(lr=4e-3),
    clip_grad=None,
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)

model = dict(
    head=dict(
        num_classes=5,
    )
)

# configure default hooks
default_hooks = dict(
    # print log every 100 iterations.
    logger=dict(type="LoggerHook", interval=10)
)

visualizer = dict(
    type="UniversalVisualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="WandbVisBackend")],
)

load_from = "checkpoints/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth"

