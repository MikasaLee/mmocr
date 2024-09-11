_base_ = [
    '../_base_/datasets/bnu_EnsExam_English.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_svtr-bnu_EnsExam_English.py',
]
# load_from = 'work_dirs/svtr_bnu_EnsExam_English_20240904_2/best_bnu_EnsExam_English_recog_AR_epoch_95.pth'
# dataset settings
train_list = [
    _base_.bnu_EnsExam_English_textrecog_train,
]

val_list = [
    _base_.bnu_EnsExam_English_textrecog_test,
]

test_list = [
    _base_.bnu_EnsExam_English_textrecog_test,
]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
valid_dataset = dict(
    type='ConcatDataset', datasets=val_list, pipeline=_base_.test_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook',
                    interval=1,
                    by_epoch=True, 
                    save_best='bnu_EnsExam_English/recog/AR',
                    save_last = True,
                    max_keep_ckpts=5,
                    rule = 'greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False),
)
train_dataloader = dict(
    batch_size=64,  # 原本是64，小一点。
    num_workers=32,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)   

val_dataloader = dict(
    batch_size=32,
    num_workers=16,  # 改回来16
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=valid_dataset)

test_dataloader = dict(
    batch_size=32,
    num_workers=16,  # 改回来16
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)


val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric'),
        dict(type='CRandARMetric')
    ],
    dataset_prefixes=['bnu_EnsExam_English'],
)   

test_evaluator = [
    dict(type='CharMetric'),
    dict(type='CRandARMetric')
]

auto_scale_lr = dict(base_batch_size=32)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

max_epochs = 100
val_interval= 1

# 令学习率为常量，即不进行学习率衰减
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        end_factor=1.,
        end=2,
        verbose=False,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=99,
        begin=2,
        end=100,
        verbose=False,
        convert_to_iter_based=True),
]

optim_wrapper = dict(  # AdamDelta不行
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        betas=(0.9, 0.99),
        eps=8e-8,
        weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# 不显示warning信息。
import warnings
warnings.filterwarnings("ignore")

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),]