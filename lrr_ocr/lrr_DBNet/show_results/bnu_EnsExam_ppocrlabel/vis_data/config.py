auto_scale_lr = dict(base_batch_size=16)
bnu_EnsExam_ppocrlabel_textdet_data_root = '/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel'
bnu_EnsExam_ppocrlabel_textdet_test = dict(
    ann_file='textdet_train.json',
    data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            1333,
            736,
        ), type='Resize'),
        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
            with_polygon=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            ),
            type='PackTextDetInputs'),
    ],
    test_mode=True,
    type='OCRDataset')
bnu_EnsExam_ppocrlabel_textdet_train = dict(
    ann_file='textdet_train.json',
    data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,
    type='OCRDataset')
bnu_EnsExam_textdet_test = dict(
    ann_file='textdet_train.json',
    data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            1333,
            736,
        ), type='Resize'),
        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
            with_polygon=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            ),
            type='PackTextDetInputs'),
    ],
    test_mode=True,
    type='OCRDataset')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=-1,
        rule='greater',
        save_best='icdar/hmean',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=True,
        draw_pred=True,
        enable=True,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/dbnet_resnet18_fpnc_1200e_ScutHccdoc/best_icdar_hmean_epoch_420.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
max_epochs = 1
model = dict(
    backbone=dict(
        depth=18,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='mmdet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TextDetDataPreprocessor'),
    det_head=dict(
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(text_repr_type='quad', type='DBPostprocessor'),
        type='DBHead'),
    neck=dict(
        in_channels=[
            64,
            128,
            256,
            512,
        ], lateral_channels=256, type='FPNC'),
    type='DBNet')
optim_wrapper = dict(
    optimizer=dict(lr=0.007, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=1,
        by_epoch=True,
        convert_to_iter_based=True,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='textdet_train.json',
        data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                736,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='HmeanIOUMetric')
test_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        736,
    ), type='Resize'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackTextDetInputs'),
]
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='textdet_train.json',
        data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                736,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
train_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        brightness=0.12549019607843137,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        args=[
            [
                'Fliplr',
                0.5,
            ],
            dict(cls='Affine', rotate=[
                -10,
                10,
            ]),
            [
                'Resize',
                [
                    0.5,
                    3.0,
                ],
            ],
        ],
        type='ImgAugWrapper'),
    dict(min_side_ratio=0.1, type='RandomCrop'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(size=(
        640,
        640,
    ), type='Pad'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
        ),
        type='PackTextDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='textdet_train.json',
        data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                736,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='HmeanIOUMetric')
val_interval = 1
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='lrr_ocr/lrr_DBNet/show_results/bnu_EnsExam_ppocrlabel',
    type='TextDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/dbnet_resnet18_fpnc_1200e_bnu_EnsExam_ppocrlabel_test'
