_base_ = [
    '../_base_/models/convnext_v2/tiny.py',
    'huangfan3_bs64_swin_384.py',
    # '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# model setting
model = dict(head=dict(num_classes=3))

# dataset setting
train_dataloader = dict(batch_size=48)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * 1024 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3/10,
        by_epoch=True,
        end=40//4,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1024)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

visualizer = dict(
    type="UniversalVisualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="WandbVisBackend")],
)

# configure default hooks
default_hooks = dict(checkpoint=dict(interval=5))

load_from = "checkpoints/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-d8579f84.pth"