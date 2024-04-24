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
auto_scale_lr = dict(base_batch_size=64)
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
load_from = 'work_dirs/nrtr_modality-transform_scut_hccdoc_20240416/best_SCUT_HccDoc_recog_AR_epoch_87.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
lr = 1
max_epochs = 100
model = dict(
    backbone=dict(height=128, type='NRTRModalityTransform'),
    data_preprocessor=dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TextRecogDataPreprocessor'),
    decoder=dict(
        dictionary=dict(
            dict_file='/lirunrui/mmocr/dicts/chinese_english_digits.txt',
            same_start_end=True,
            type='Dictionary',
            with_end=True,
            with_padding=True,
            with_start=True,
            with_unknown=True),
        max_seq_len=30,
        module_loss=dict(
            flatten=True, ignore_first_char=True, type='CEModuleLoss'),
        postprocessor=dict(type='AttentionPostprocessor'),
        type='NRTRDecoder'),
    encoder=dict(is_PE=True, n_layers=12, type='NRTREncoder'),
    type='NRTR')
optim_wrapper = dict(
    optimizer=dict(eps=1e-05, lr=1, type='Adadelta'), type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=100,
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
    batch_size=32,
    dataset=dict(
        datasets=[
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
                height=128,
                max_width=2304,
                min_width=576,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='PadToWidth', width=2304),
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
            ann_file='textrecog_train.json',
            data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            height=128,
            max_width=2304,
            min_width=576,
            type='RescaleToHeight',
            width_divisor=16),
        dict(type='PadToWidth', width=2304),
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
test_evaluator = dict(
    dataset_prefixes=[
        'bnu_EnsExam_ppocrlabel',
    ],
    metrics=[
        dict(type='CharMetric'),
        dict(type='CRandARMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_list = [
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
        height=128,
        max_width=2304,
        min_width=576,
        type='RescaleToHeight',
        width_divisor=16),
    dict(type='PadToWidth', width=2304),
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
            dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                height=128,
                max_width=2304,
                min_width=576,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='PadToWidth', width=2304),
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
        dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            height=128,
            max_width=2304,
            min_width=576,
            type='RescaleToHeight',
            width_divisor=16),
        dict(type='PadToWidth', width=2304),
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
    dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        height=128,
        max_width=2304,
        min_width=576,
        type='RescaleToHeight',
        width_divisor=16),
    dict(type='PadToWidth', width=2304),
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
                dict(
                    height=32,
                    max_width=160,
                    min_width=32,
                    type='RescaleToHeight',
                    width_divisor=16),
            ],
            [
                dict(type='PadToWidth', width=160),
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
                ann_file='textrecog_train.json',
                data_root='/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                height=128,
                max_width=2304,
                min_width=576,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='PadToWidth', width=2304),
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
        'bnu_EnsExam_ppocrlabel',
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
    save_dir='lrr_ocr/lrr_NRTR/show_results/bnu_EnsExam_ppocrlabel_with_PE',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/nrtr_modality-transform_6e_bnu_EnsExam_ppocrlabel_test'
