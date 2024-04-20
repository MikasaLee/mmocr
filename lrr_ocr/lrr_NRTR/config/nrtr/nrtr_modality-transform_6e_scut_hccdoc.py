_base_ = [
    '_base_nrtr_modality-transform_chinese.py',
    '../_base_/datasets/scut_hccdoc.py',
    '../_base_/datasets/CASIA_HWDB_chineseocr_data.py',
    '../_base_/datasets/CASIA_HWDB_official_2x_data.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
]


# dataset settings

train_list = [
    _base_.CASIA_HWDB_chineseocr_data_textrecog_train,
    _base_.CASIA_HWDB_official_2x_data_textrecog_train,
    _base_.CASIA_HWDB_official_2x_data_textrecog_val,
    _base_.scut_hccdoc_textrecog_train,
]

test_list = [
    _base_.CASIA_HWDB_official_2x_data_textrecog_test,
    _base_.scut_hccdoc_textrecog_test,
]


train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook',
                    interval=1,
                    by_epoch=True, 
                    save_best='SCUT_HccDoc/recog/AR',
                    save_last = True,
                    max_keep_ckpts=5,
                    rule = 'greater'),
)
 
train_dataloader = dict(
    batch_size=64,
    num_workers=32,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)   

val_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        # dict(
        #     type='WordMetric',
        #     mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric'),
        dict(type='CRandARMetric')
    ],
    dataset_prefixes=['Casia_Hwdb_2x','SCUT_HccDoc'])   # 这个顺序要和 test_list 一致，要不然就乱了

test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

max_epochs = 100
val_interval= 1
lr = 1

# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, convert_to_iter_based=True,eta_min=0.1)]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adadelta', lr=lr, eps=1e-5))  # 必须用Adadelta

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# 不显示warning信息。
import warnings
warnings.filterwarnings("ignore")

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),]

