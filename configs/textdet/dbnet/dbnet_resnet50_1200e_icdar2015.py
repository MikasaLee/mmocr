_base_ = [
    'dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
]

load_from = None

_base_.model.backbone = dict(
    type='mmdet.ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.002

# 分别测试IOU的阈值在0.5,0.6,0.7,0.8的性能。
val_evaluator = dict(type='HmeanIOUMetric', pred_score_thrs=dict(start=0.5, stop=0.8, step=0.1))
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,    # 他这边的这个意思是不做验证的，训练时epoch数满足了interval就会保存，这个和下面的train_cfg.val_interval建议统一
        by_epoch=True,
        max_keep_ckpts=5,
        save_last = True,
        save_best = 'icdar/hmean',  # 前面icdar是写死了，见mmocr/evaluation/metrics/hmean_iou_metric.py文件。感觉应该是官方拿第一次测试完之后后面没改过来。
        rule = 'greater',
    ))

param_scheduler = [
    dict(type='LinearLR', end=100, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=100, end=1200),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1200, val_interval= 1 )