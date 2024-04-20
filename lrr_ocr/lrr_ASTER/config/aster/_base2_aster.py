dictionary = dict(
    type='Dictionary',
    dict_file='/lirunrui/mmocr/dicts/chinese_english_digits.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='ASTER',
    preprocessor=dict(
        type='STN',
        in_channels=1,
        resized_image_size=(128, 576),
        output_image_size=(128, 576),
        num_control_points=20),
    backbone=dict(type='ScutHccdocBackbone', leaky_relu=False, input_channels=1),
    encoder=dict(type='ASTEREncoder', in_channels=512),
    decoder=dict(
        type='ASTERDecoder',
        max_seq_len=25,
        in_channels=512,
        emb_dims=512,
        attn_dims=512,
        hidden_size=512,
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', flatten=True, ignore_first_char=True),
        # module_loss=dict(type='CTCModuleLoss', letter_case='lower',zero_infinity=True),
        # postprocessor=dict(type='CTCPostProcessor'),
        dictionary=dictionary,
    ),

    data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[127], std=[127]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='grayscale',
        ignore_empty=True,
        min_size=2),

    dict(
        type='RescaleToHeight',
        height=128,
        min_width=576,
        max_width=None,
        width_divisor=16), 
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='RescaleToHeight',
        height=128,
        min_width=576,
        max_width=None,
        width_divisor=16),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

tta_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=0, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=1, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=3, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
            ],
            [
                dict(
                    type='RescaleToHeight',
                    height=128,
                    min_width=576,
                    max_width=None,
                    width_divisor=16),
            ],
            # add loading annotation after ``Resize`` because ground truth
            # does not need to do resize data transform
            [dict(type='LoadOCRAnnotations', with_text=True)],
            [
                dict(
                    type='PackTextRecogInputs',
                    meta_keys=('img_path', 'ori_shape', 'img_shape',
                               'valid_ratio'))
            ]
        ])
]
