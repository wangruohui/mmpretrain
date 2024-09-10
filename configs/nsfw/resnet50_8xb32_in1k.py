_base_ = [
    "../_base_/models/resnet50.py",
    "nsfw_bs32_336.py",
    "../_base_/schedules/imagenet_bs256.py",
    "../_base_/default_runtime.py",
]

model = dict(head=dict(num_classes=5))

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))

# learning policy
param_scheduler = dict(
    type="MultiStepLR", by_epoch=True, milestones=[10,], gamma=0.1
)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)

# runtime settings
load_from = "checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth"

# configure default hooks
default_hooks = dict(
    # print log every 100 iterations.
    logger=dict(type="LoggerHook", interval=50)
)

visualizer = dict(
    type="UniversalVisualizer",
    # vis_backends=[dict(type="LocalVisBackend")],
    vis_backends=[dict(type="LocalVisBackend"), dict(type="WandbVisBackend")],
)
