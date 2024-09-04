CASIA_HWDB_chineseocr_data_textrecog_data_root = '/lirunrui/datasets/mmocr_CASIA_HWDB_chinese_ocr_dataset'
CASIA_HWDB_chineseocr_data_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/lirunrui/datasets/mmocr_CASIA_HWDB_chinese_ocr_dataset',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
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
auto_scale_lr = dict(base_batch_size=32)
bnu_EnsExam_ppocrlabel_textrecog_data_root = '/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel'
bnu_EnsExam_ppocrlabel_textrecog_test = dict(
    ann_file='textrecog_train.json',
    data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
bnu_EnsExam_ppocrlabel_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
    pipeline=None,
    type='OCRDataset')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        rule='greater',
        save_best='bnu_EnsExam_ppocrlabel/recog/AR',
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
        font_properties='/usr/share/fonts/fonts_library/simsun.ttc',
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
load_from = 'work_dirs/svtr_HWCR_add_scutept_20240828/best_bnu_EnsExam_ppocrlabel_recog_AR_epoch_68.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
lr = 1
max_epochs = 100
model = dict(
    data_preprocessor=dict(
        mean=[
            127.5,
        ], std=[
            127.5,
        ], type='TextRecogDataPreprocessor'),
    decoder=dict(
        dictionary=dict(
            dict_file='/lirunrui/mmocr/dicts/chinese_english_digits.txt',
            same_start_end=True,
            type='Dictionary',
            with_end=True,
            with_padding=True,
            with_start=True,
            with_unknown=True),
        in_channels=256,
        max_seq_len=180,
        module_loss=dict(
            letter_case='lower', type='CTCModuleLoss', zero_infinity=True),
        postprocessor=dict(type='CTCPostProcessor'),
        type='SVTRDecoder'),
    encoder=dict(
        depth=[
            3,
            6,
            9,
        ],
        embed_dims=[
            128,
            256,
            384,
        ],
        img_size=[
            32,
            576,
        ],
        in_channels=3,
        max_seq_len=40,
        merging_types='Conv',
        mixer_types=[
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Local',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
            'Global',
        ],
        num_heads=[
            4,
            8,
            12,
        ],
        out_channels=256,
        prenorm=False,
        type='SVTREncoder',
        window_size=[
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
            [
                7,
                11,
            ],
        ]),
    type='SVTR')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.99,
        ),
        eps=8e-08,
        lr=0.0005,
        type='AdamW',
        weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        convert_to_iter_based=True,
        end=2,
        end_factor=1.0,
        start_factor=0.5,
        type='LinearLR',
        verbose=False),
    dict(
        T_max=19,
        begin=2,
        convert_to_iter_based=True,
        end=20,
        type='CosineAnnealingLR',
        verbose=False),
]
randomness = dict(seed=None)
resume = False
scut_ept_textrecog_data_root = '/lirunrui/datasets/mmocr_SCUT_EPT_Dataset'
scut_ept_textrecog_test = dict(
    ann_file='textrecog_test.json',
    data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
scut_ept_textrecog_train = dict(
    ann_file='textrecog_train.json',
    data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
    pipeline=None,
    type='OCRDataset')
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
    batch_size=32,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
            dict(
                ann_file='textrecog_train.json',
                data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                height=32,
                max_width=576,
                min_width=32,
                type='RescaleToHeight',
                width_divisor=4),
            dict(type='PadToWidth', width=576),
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
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset = dict(
    datasets=[
        dict(
            ann_file='textrecog_test.json',
            data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
        dict(
            ann_file='textrecog_train.json',
            data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            height=32,
            max_width=576,
            min_width=32,
            type='RescaleToHeight',
            width_divisor=4),
        dict(type='PadToWidth', width=576),
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
    type='ConcatDataset')
test_evaluator = [
    dict(
        dataset_prefixes=[
            'SCUT_EPT',
            'bnu_EnsExam_ppocrlabel',
        ],
        metrics=[
            dict(type='CharMetric'),
            dict(type='CRandARMetric'),
        ],
        type='MultiDatasetsEvaluator'),
    dict(
        out_file_path=
        './work_dirs/svtr-base_HWCR/best_bnu_EnsExam_ppocrlabel_recog_AR_epoch_68.pth_predictions.pkl',
        type='DumpResults'),
]
test_list = [
    dict(
        ann_file='textrecog_test.json',
        data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
    dict(
        ann_file='textrecog_train.json',
        data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        height=32,
        max_width=576,
        min_width=32,
        type='RescaleToHeight',
        width_divisor=4),
    dict(type='PadToWidth', width=576),
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
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_train.json',
                data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
                pipeline=None,
                type='OCRDataset'),
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
            dict(
                height=32,
                max_width=576,
                min_width=32,
                type='RescaleToHeight',
                width_divisor=4),
            dict(type='PadToWidth', width=576),
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
            data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
            pipeline=None,
            type='OCRDataset'),
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
        dict(
            height=32,
            max_width=576,
            min_width=32,
            type='RescaleToHeight',
            width_divisor=4),
        dict(type='PadToWidth', width=576),
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
        data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
        pipeline=None,
        type='OCRDataset'),
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
    dict(
        height=32,
        max_width=576,
        min_width=32,
        type='RescaleToHeight',
        width_divisor=4),
    dict(type='PadToWidth', width=576),
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
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        datasets=[
            dict(
                ann_file='textrecog_test.json',
                data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
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
            dict(
                ann_file='textrecog_train.json',
                data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                height=32,
                max_width=576,
                min_width=32,
                type='RescaleToHeight',
                width_divisor=4),
            dict(type='PadToWidth', width=576),
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
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'SCUT_EPT',
        'Casia_Hwdb_2x',
        'SCUT_HccDoc',
        'bnu_EnsExam_ppocrlabel',
    ],
    metrics=[
        dict(type='CharMetric'),
        dict(type='CRandARMetric'),
    ],
    type='MultiDatasetsEvaluator')
val_interval = 1
val_list = [
    dict(
        ann_file='textrecog_test.json',
        data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
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
    dict(
        ann_file='textrecog_train.json',
        data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
valid_dataset = dict(
    datasets=[
        dict(
            ann_file='textrecog_test.json',
            data_root='/lirunrui/datasets/mmocr_SCUT_EPT_Dataset',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
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
        dict(
            ann_file='textrecog_train.json',
            data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            height=32,
            max_width=576,
            min_width=32,
            type='RescaleToHeight',
            width_divisor=4),
        dict(type='PadToWidth', width=576),
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
    type='ConcatDataset')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='lrr_ocr/lrr_SVTR/show_results_add_scutept',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/svtr-base_HWCR'
