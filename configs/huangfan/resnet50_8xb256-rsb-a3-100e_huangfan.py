_base_ = [
    '../_base_/models/resnet50.py',
    'huangfan_bs256_rsb_a3.py',
    '../_base_/schedules/imagenet_bs2048_rsb.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
    head=dict(num_classes=3, loss=dict(use_sigmoid=True)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ]),
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(lr=0.008/8),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

# finetune
load_from = 'checkpoints/resnet50_8xb256-rsb-a3-100e_in1k_20211228-3493673c.pth'

default_hooks = dict(checkpoint=dict(interval=5))

visualizer = dict(
    type="UniversalVisualizer",
    # vis_backends=[dict(type="LocalVisBackend")],
    vis_backends=[dict(type="LocalVisBackend"), dict(type="WandbVisBackend")],
)

# train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=5)
