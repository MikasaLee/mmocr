auto_scale_lr = dict(base_batch_size=4)
bnu_EnsExam_ppocrlabel_textdet_data_root = '/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel'
bnu_EnsExam_ppocrlabel_textdet_test = dict(
    ann_file='textdet_train.json',
    data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            1920,
            1920,
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
            1920,
            1920,
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
        max_keep_ckpts=10,
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
load_from = 'work_dirs/mask-rcnn_resnet50_fpn_160e_ScutHccdoc/best_icdar_hmean_epoch_254.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
lr = 0.008
mask_rcnn = dict(
    _scope_='mmdet',
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=1,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.17,
                0.44,
                1.13,
                2.9,
                7.46,
            ],
            scales=[
                4,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='MaskRCNN')
max_epochs = 1200
model = dict(
    cfg=dict(
        _scope_='mmdet',
        backbone=dict(
            depth=50,
            frozen_stages=1,
            init_cfg=dict(
                checkpoint='torchvision://resnet50', type='Pretrained'),
            norm_cfg=dict(requires_grad=True, type='BN'),
            norm_eval=True,
            num_stages=4,
            out_indices=(
                0,
                1,
                2,
                3,
            ),
            style='pytorch',
            type='ResNet'),
        data_preprocessor=dict(
            bgr_to_rgb=True,
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            pad_mask=False,
            pad_size_divisor=32,
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type='DetDataPreprocessor'),
        neck=dict(
            in_channels=[
                256,
                512,
                1024,
                2048,
            ],
            num_outs=5,
            out_channels=256,
            type='FPN'),
        roi_head=dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=1,
                reg_class_agnostic=False,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            mask_head=dict(
                conv_out_channels=256,
                in_channels=256,
                loss_mask=dict(
                    loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
                num_classes=1,
                num_convs=4,
                type='FCNMaskHead'),
            mask_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=14, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            type='StandardRoIHead'),
        rpn_head=dict(
            anchor_generator=dict(
                ratios=[
                    0.17,
                    0.44,
                    1.13,
                    2.9,
                    7.46,
                ],
                scales=[
                    4,
                ],
                strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                type='AnchorGenerator'),
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
            type='RPNHead'),
        test_cfg=dict(
            rcnn=dict(
                mask_thr_binary=0.5,
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.05),
            rpn=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=1000)),
        train_cfg=dict(
            rcnn=dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
            rpn_proposal=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=2000)),
        type='MaskRCNN'),
    text_repr_type='poly',
    type='MMDetWrapper')
optim_wrapper = dict(
    optimizer=dict(lr=0.008, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=1200,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=8e-06,
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
                1920,
                1920,
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
test_evaluator = dict(
    pred_score_thrs=dict(start=0.5, step=0.1, stop=0.8), type='HmeanIOUMetric')
test_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1920,
        1920,
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
train_cfg = dict(max_epochs=1200, type='EpochBasedTrainLoop', val_interval=1)
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
                1920,
                1920,
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
        contrast=0.5,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        keep_ratio=True,
        ratio_range=(
            1.0,
            4.125,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(target_size=(
        640,
        640,
    ), type='TextDetRandomCrop'),
    dict(poly2mask=True, type='MMOCR2MMDet'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'scale_factor',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
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
                1920,
                1920,
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
val_evaluator = dict(
    pred_score_thrs=dict(start=0.5, step=0.1, stop=0.8), type='HmeanIOUMetric')
val_interval = 1
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir=
    'lrr_ocr/lrr_maskrcnn/show_results/bnu_EnsExam_ppocrlabel_resnet50',
    type='TextDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/mask-rcnn_resnet50_fpn_160e_bnu_EnsExam_ppocrlabel_test'
