CASIA_HWDB_chineseocr_data_textrecog_data_root = '/lirunrui/datasets/mmocr_CASIA_HWDB_chinese_ocr_dataset'
CASIA_HWDB_chineseocr_data_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_chinese_ocr_dataset',
    pipeline=None,
    type='OCRDataset')
CASIA_HWDB_official_2x_data_textrecog_data_root = '/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset'
CASIA_HWDB_official_2x_data_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
CASIA_HWDB_official_2x_data_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
    pipeline=None,
    type='OCRDataset')
CASIA_HWDB_official_2x_data_textrecog_val = dict(
    ann_file='textrecog_val.json',
    data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
    pipeline=None,
    type='OCRDataset')
auto_scale_lr = dict(base_batch_size=64)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        rule='greater',
        save_best='SCUT_HccDoc/recog/AR',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
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
dictionary = dict(
    dict_file='/lirunrui/mmocr/dicts/chinese_english_digits.txt',
    same_start_end=True,
    type='Dictionary',
    with_end=True,
    with_padding=True,
    with_start=True,
    with_unknown=True)
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/aster_resnet45_6e_scut_hccdoc/best_SCUT_HccDoc_recog_AR_epoch_894.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
lr = 1
max_epochs = 1200
model = dict(
    backbone=dict(
        arch_channels=[
            32,
            64,
            128,
            256,
            512,
        ],
        arch_layers=[
            3,
            4,
            6,
            6,
            3,
        ],
        block_cfgs=dict(type='BasicBlock', use_conv1x1='True'),
        in_channels=3,
        init_cfg=[
            dict(layer='Conv2d', type='Kaiming'),
            dict(layer='BatchNorm2d', type='Constant', val=1),
        ],
        stem_channels=[
            32,
        ],
        strides=[
            (
                2,
                2,
            ),
            (
                2,
                2,
            ),
            (
                2,
                1,
            ),
            (
                2,
                1,
            ),
            (
                2,
                1,
            ),
        ],
        type='ResNet'),
    data_preprocessor=dict(
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='TextRecogDataPreprocessor'),
    decoder=dict(
        attn_dims=512,
        dictionary=dict(
            dict_file='/lirunrui/mmocr/dicts/chinese_english_digits.txt',
            same_start_end=True,
            type='Dictionary',
            with_end=True,
            with_padding=True,
            with_start=True,
            with_unknown=True),
        emb_dims=512,
        hidden_size=512,
        in_channels=512,
        max_seq_len=25,
        module_loss=dict(
            flatten=True, ignore_first_char=True, type='CEModuleLoss'),
        postprocessor=dict(type='AttentionPostprocessor'),
        type='ASTERDecoder'),
    encoder=dict(in_channels=512, type='ASTEREncoder'),
    preprocessor=dict(
        in_channels=3,
        num_control_points=20,
        output_image_size=(
            32,
            100,
        ),
        resized_image_size=(
            32,
            64,
        ),
        type='STN'),
    type='ASTER')
optim_wrapper = dict(
    optimizer=dict(eps=1e-05, lr=1, type='Adadelta'), type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=1200,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0.1,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=None)
resume = False
scut_hccdoc_textrecog_data_root = '/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset'
scut_hccdoc_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
scut_hccdoc_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
    pipeline=None,
    type='OCRDataset')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root=
                '/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
            dict(
                ann_file='textrecog_test.json',
                data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                    'instances',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset = dict(
    datasets=[
        dict(
            ann_file='textrecog_test.json',
            data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
        dict(
            ann_file='textrecog_test.json',
            data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(scale=(
            256,
            64,
        ), type='Resize'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'valid_ratio',
                'instances',
            ),
            type='PackTextRecogInputs'),
    ],
    type='ConcatDataset')
test_evaluator = dict(
    dataset_prefixes=[
        'Casia_Hwdb_2x',
        'SCUT_HccDoc',
    ],
    metrics=[
        dict(type='CharMetric'),
        dict(type='CRandARMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_list = [
    dict(
        ann_file='textrecog_test.json',
        data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_test.json',
        data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
            'instances',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=1200, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_train.json',
                data_root=
                '/lirunrui/datasets/mmocr_CASIA_HWDB_chinese_ocr_dataset',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='textrecog_train.json',
                data_root=
                '/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='textrecog_val.json',
                data_root=
                '/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
                pipeline=None,
                type='OCRDataset'),
            dict(
                ann_file='textrecog_train.json',
                data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    num_workers=32,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    datasets=[
        dict(
            ann_file='textrecog_train.json',
            data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_chinese_ocr_dataset',
            pipeline=None,
            type='OCRDataset'),
        dict(
            ann_file='textrecog_train.json',
            data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
            pipeline=None,
            type='OCRDataset'),
        dict(
            ann_file='textrecog_val.json',
            data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
            pipeline=None,
            type='OCRDataset'),
        dict(
            ann_file='textrecog_train.json',
            data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
            pipeline=None,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(scale=(
            256,
            64,
        ), type='Resize'),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'valid_ratio',
            ),
            type='PackTextRecogInputs'),
    ],
    type='ConcatDataset')
train_list = [
    dict(
        ann_file='textrecog_train.json',
        data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_chinese_ocr_dataset',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_val.json',
        data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
        pipeline=None,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(ignore_empty=True, min_size=5, type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(scale=(
        256,
        64,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
            ],
            [
                dict(scale=(
                    256,
                    64,
                ), type='Resize'),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                        'instances',
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root=
                '/lirunrui/datasets/mmocr_CASIA_HWDB_official_2x_dataset',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
            dict(
                ann_file='textrecog_test.json',
                data_root='/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                256,
                64,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                    'instances',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'Casia_Hwdb_2x',
        'SCUT_HccDoc',
    ],
    metrics=[
        dict(type='CharMetric'),
        dict(type='CRandARMetric'),
    ],
    type='MultiDatasetsEvaluator')
val_interval = 1
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='lrr_ocr/lrr_ASTER/show_results',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/aster_resnet45_6e_scut_hccdoc'
