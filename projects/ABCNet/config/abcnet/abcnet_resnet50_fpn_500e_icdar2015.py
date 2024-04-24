_base_ = [
    '_base_abcnet_resnet50_fpn.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_sgd_500e.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
]

# dataset settings
icdar2015_textspotting_train = _base_.icdar2015_textspotting_train
icdar2015_textspotting_train.pipeline = _base_.train_pipeline
icdar2015_textspotting_test = _base_.icdar2015_textspotting_test
icdar2015_textspotting_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=icdar2015_textspotting_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=icdar2015_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

max_epochs = 500
val_interval= 1
lr = 0.1

# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, convert_to_iter_based=True,eta_min=lr*0.01)]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adadelta', lr=lr, eps=1e-5))  # 必须用Adadelta

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

custom_imports = dict(imports=['abcnet'], allow_failed_imports=False)

load_from = 'https://download.openmmlab.com/mmocr/textspotting/abcnet/abcnet_resnet50_fpn_500e_icdar2015/abcnet_resnet50_fpn_pretrain-d060636c.pth'  # noqa

find_unused_parameters = True
