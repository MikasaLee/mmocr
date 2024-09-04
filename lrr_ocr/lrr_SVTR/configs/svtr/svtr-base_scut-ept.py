# 基于svtr-base-HWCR文件改的
# 复现scut-ept的实验结果，主要是他这在18年的时候CR、AR的指标已经达到了80/75,这已经很高了，主要我看区别是尺寸设置为了(96,1440)
# 只用scut-ept的训练集训了一下，不行完全不拟合，paper中还说它们还通过CASIA-HWDB生成了一个18w的文本行进行训练。
_base_ = [
    '../_base_/datasets/scut_ept.py',   # new add
    '../_base_/datasets/scut_hccdoc.py',
    '../_base_/datasets/CASIA_HWDB_chineseocr_data.py',
    '../_base_/datasets/CASIA_HWDB_official_2x_data.py',
    '../_base_/datasets/bnu_EnsExam_ppocrlabel.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_svtr-scut_ept.py',
]

# load_from = 'work_dirs/svtr_HWCR_20240822/best_bnu_EnsExam_ppocrlabel_recog_AR_epoch_18.pth'
# dataset settings
train_list = [
    _base_.scut_ept_textrecog_train,
    # _base_.CASIA_HWDB_chineseocr_data_textrecog_train,
    # _base_.CASIA_HWDB_official_2x_data_textrecog_train,
    # _base_.CASIA_HWDB_official_2x_data_textrecog_val,
    # _base_.scut_hccdoc_textrecog_train,
]

val_list = [
    _base_.scut_ept_textrecog_test,
    # _base_.CASIA_HWDB_official_2x_data_textrecog_test,
    # _base_.scut_hccdoc_textrecog_test,
    _base_.bnu_EnsExam_ppocrlabel_textrecog_test,
]

test_list = [
    _base_.bnu_EnsExam_ppocrlabel_textrecog_test,
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
                    save_best='bnu_EnsExam_ppocrlabel/recog/AR',
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
    batch_size=4,  # 原本是64，小一点。
    num_workers=32,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)   

val_dataloader = dict(
    batch_size=4,
    num_workers=16,  # 改回来16
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=valid_dataset)

test_dataloader = dict(
    batch_size=4,
    num_workers=16,  # 改回来16
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)


val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        # dict(
        #     type='WordMetric',
        #     mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric'),
        dict(type='CRandARMetric')
    ],
    # dataset_prefixes=['SCUT_EPT','Casia_Hwdb_2x','SCUT_HccDoc','bnu_EnsExam_ppocrlabel'])   # 这个顺序要和 test_list 一致，要不然就乱
    dataset_prefixes=['SCUT_EPT','bnu_EnsExam_ppocrlabel'])   # 这个顺序要和 test_list 一致，要不然就乱

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
lr = 1

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
        T_max=19,
        begin=2,
        end=20,
        verbose=False,
        convert_to_iter_based=True),
]

optim_wrapper = dict(  # AdamDelta不行
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5 / (10**4) * 2048 / 2048,
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


model = dict(
    # preprocessor=dict(output_image_size=(32, 576), ),
    encoder=dict(
        img_size=[96, 1440],
        max_seq_len=40,
        out_channels=256,
        embed_dims=[128, 256, 384],
        depth=[3, 6, 9],
        num_heads=[4, 8, 12],
        mixer_types=['Local'] * 8 + ['Global'] * 10),
    decoder=dict(in_channels=256))