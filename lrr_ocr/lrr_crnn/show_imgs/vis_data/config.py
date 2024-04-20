auto_scale_lr = dict(base_batch_size=256)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        rule='greater',
        save_best='ignore_case_symbol',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
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
    dict_file=
    '/lirunrui/mmocr/lrr_ocr/lrr_CRNN/config/crnn/../../dicts/lower_english_digits.txt',
    type='Dictionary',
    with_padding=True)
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
icdar2015_1811_textrecog_test = dict(
    ann_file='textrecog_test_1811.json',
    data_root='/lirunrui/datasets/icdar2015_textrecog',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2015_textrecog_data_root = '/lirunrui/datasets/icdar2015_textrecog'
icdar2015_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/lirunrui/datasets/icdar2015_textrecog',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
icdar2015_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/lirunrui/datasets/icdar2015_textrecog',
    pipeline=None,
    type='OCRDataset')
launcher = 'none'
load_from = 'work_dirs/crnn_mini-vgg_5e_mj/epoch_5.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
mjsynth_sub_textrecog_train = dict(
    ann_file='subset_textrecog_train.json',
    data_root='/lirunrui/datasets/mjsynth',
    pipeline=None,
    type='OCRDataset')
mjsynth_textrecog_data_root = '/lirunrui/datasets/mjsynth'
mjsynth_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/lirunrui/datasets/mjsynth',
    pipeline=None,
    type='OCRDataset')
model = dict(
    backbone=dict(input_channels=1, leaky_relu=False, type='MiniVGG'),
    data_preprocessor=dict(
        mean=[
            127,
        ], std=[
            127,
        ], type='TextRecogDataPreprocessor'),
    decoder=dict(
        dictionary=dict(
            dict_file=
            '/lirunrui/mmocr/lrr_ocr/lrr_CRNN/config/crnn/../../dicts/lower_english_digits.txt',
            type='Dictionary',
            with_padding=True),
        in_channels=512,
        module_loss=dict(letter_case='lower', type='CTCModuleLoss'),
        postprocessor=dict(type='CTCPostProcessor'),
        rnn_flag=True,
        type='CRNNDecoder'),
    encoder=None,
    preprocessor=None,
    type='CRNN')
optim_wrapper = dict(
    optimizer=dict(lr=1.0, type='Adadelta'), type='OptimWrapper')
param_scheduler = [
    dict(factor=1.0, type='ConstantLR'),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='/lirunrui/datasets/icdar2015_textrecog',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(color_type='grayscale', type='LoadImageFromFile'),
            dict(
                height=32,
                max_width=None,
                min_width=32,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='LoadOCRAnnotations', with_text=True),
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
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    dataset_prefixes=[
        'IC15',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_list = [
    dict(
        ann_file='textrecog_test.json',
        data_root='/lirunrui/datasets/icdar2015_textrecog',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(color_type='grayscale', type='LoadImageFromFile'),
    dict(
        height=32,
        max_width=None,
        min_width=32,
        type='RescaleToHeight',
        width_divisor=16),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=5, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_train.json',
                data_root='/lirunrui/datasets/mjsynth',
                pipeline=None,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(
                color_type='grayscale',
                ignore_empty=True,
                min_size=2,
                type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(keep_ratio=False, scale=(
                100,
                32,
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
    num_workers=24,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_list = [
    dict(
        ann_file='textrecog_train.json',
        data_root='/lirunrui/datasets/mjsynth',
        pipeline=None,
        type='OCRDataset'),
]
train_pipeline = [
    dict(
        color_type='grayscale',
        ignore_empty=True,
        min_size=2,
        type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(keep_ratio=False, scale=(
        100,
        32,
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
    dict(color_type='grayscale', type='LoadImageFromFile'),
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
                dict(
                    height=32,
                    max_width=None,
                    min_width=32,
                    type='RescaleToHeight',
                    width_divisor=16),
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
                data_root='/lirunrui/datasets/icdar2015_textrecog',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(color_type='grayscale', type='LoadImageFromFile'),
            dict(
                height=32,
                max_width=None,
                min_width=32,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='LoadOCRAnnotations', with_text=True),
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
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'IC15',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='lrr_ocr/lrr_crnn/show_imgs',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/crnn_mini-vgg_5e_mj'
