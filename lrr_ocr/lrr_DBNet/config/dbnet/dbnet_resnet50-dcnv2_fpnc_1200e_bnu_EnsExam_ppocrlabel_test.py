_base_ = [
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    '../_base_/datasets/bnu_EnsExam_ppocrlabel.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_polygon=True,
        with_label=True,
    ),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='ImgAugWrapper',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]


test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2560, 1440), keep_ratio=True),  #原scale=(4068, 1024)，感觉这个太大了，没必要。
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


# dataset settings
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



auto_scale_lr = dict(base_batch_size=12)

max_epochs = 1200
val_interval= 1
lr = 0.01


# 每 10 个 epoch 储存一次权重，且只保留最后一个权重
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=val_interval,    # 他这边的这个意思是不做验证的，训练时epoch数满足了interval就会保存，这个和下面的train_cfg.val_interval建议统一
        by_epoch=True,
        max_keep_ckpts=10,
        save_last = True,
        save_best = 'icdar/hmean',  # 前面icdar是写死了，见mmocr/evaluation/metrics/hmean_iou_metric.py文件。感觉应该是官方拿第一次测试完之后后面没改过来。
        rule = 'greater',
    ))

# 分别测试IOU的阈值在0.5,0.6,0.7,0.8的性能。
val_evaluator = dict(type='HmeanIOUMetric', pred_score_thrs=dict(start=0.5, stop=0.8, step=0.1))
test_evaluator = val_evaluator



# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, convert_to_iter_based=True,eta_min=lr*0.01)]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval= val_interval )