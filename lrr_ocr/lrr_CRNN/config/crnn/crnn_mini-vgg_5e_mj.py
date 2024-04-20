# training schedule for 1x
_base_ = [
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]
# dataset settings
train_list = [_base_.mjsynth_textrecog_train]
test_list = [ _base_.icdar2015_textrecog_test]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),    #interval 控制的是多少个step输出。
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook',
                    interval=1,
                    by_epoch=True, save_best='ignore_case_symbol',
                    rule = 'greater'),
)

train_dataloader = dict(
    batch_size=64,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))
test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))
val_dataloader = test_dataloader

val_evaluator = dict(
    dataset_prefixes=['IC15'])
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64 * 4)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 不显示warning信息。
import warnings
warnings.filterwarnings("ignore")
