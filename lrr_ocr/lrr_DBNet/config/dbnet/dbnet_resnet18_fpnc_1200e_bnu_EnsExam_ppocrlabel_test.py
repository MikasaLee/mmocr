_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/bnu_EnsExam_ppocrlabel.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]


bnu_EnsExam_textdet_test = _base_.bnu_EnsExam_ppocrlabel_textdet_test
bnu_EnsExam_textdet_test.pipeline = _base_.test_pipeline

val_dataloader = dict(
    batch_size=16,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=bnu_EnsExam_textdet_test)

train_dataloader = val_dataloader
test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)

max_epochs = 1
val_interval= int(max(max_epochs*0.05,1))


# 每 10 个 epoch 储存一次权重，且只保留最后一个权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=val_interval,    # 这个和下面的train_cfg.val_interval建议统一
        by_epoch=True,
        max_keep_ckpts=-1,
        save_last = True,
        save_best = 'icdar/hmean',  # 前面icdar是写死了，见mmocr/evaluation/metrics/hmean_iou_metric.py文件。感觉应该是官方拿第一次测试完之后后面没改过来。
        rule = 'greater',
    ))

 
# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, convert_to_iter_based=True)]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001))

# 设置最大 epoch 数为 max_epochs; max_epochs*0.05 个 epoch 运行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval= val_interval )
