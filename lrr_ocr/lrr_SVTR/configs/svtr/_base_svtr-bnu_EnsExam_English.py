dictionary = dict(
    type='Dictionary',
    dict_file='/lirunrui/mmocr/dicts/english_digits_symbols_space.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='SVTR',
    encoder=dict(
        type='SVTREncoder',
        img_size=[32, 576],
        in_channels=3,
        out_channels=256,
        embed_dims=[128, 256, 384],
        depth=[3, 6, 9],
        num_heads=[4, 8, 12],
        mixer_types=['Local'] * 8 + ['Global'] * 10,
        window_size=[[7, 11], [7, 11], [7, 11]],
        merging_types='Conv',
        prenorm=False,
        max_seq_len=220),
    decoder=dict(
        type='SVTRDecoder',
        in_channels=256,
        max_seq_len=220,
        # module_loss=dict(
        #     type='CTCModuleLoss', letter_case='lower', zero_infinity=True),
        module_loss=dict(  # 关掉小写
            type='CTCModuleLoss', zero_infinity=True),
        postprocessor=dict(type='CTCPostProcessor'),
        dictionary=dictionary),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[127.5], std=[127.5]))

train_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True, min_size=5),
    dict(type='LoadOCRAnnotations', with_text=True),
    # dict(
    #     type='RandomApply',
    #     prob=0.4,
    #     transforms=[
    #         dict(type='TextRecogGeneralAug', ),
    #     ],
    # ),
    # dict(
    #     type='RandomApply',
    #     prob=0.4,
    #     transforms=[
    #         dict(type='CropHeight', ),
    #     ],
    # ),
    # dict(
    #     type='ConditionApply',
    #     condition='min(results["img_shape"])>10',
    #     true_transforms=dict(
    #         type='RandomApply',
    #         prob=0.4,
    #         transforms=[
    #             dict(
    #                 type='TorchVisionWrapper',
    #                 op='GaussianBlur',
    #                 kernel_size=5,
    #                 sigma=1,
    #             ),
    #         ],
    #     )),
    # dict(
    #     type='RandomApply',
    #     prob=0.4,
    #     transforms=[
    #         dict(
    #             type='TorchVisionWrapper',
    #             op='ColorJitter',
    #             brightness=0.5,
    #             saturation=0.5,
    #             contrast=0.5,
    #             hue=0.1),
    #     ]),
    # dict(
    #     type='RandomApply',
    #     prob=0.4,
    #     transforms=[
    #         dict(type='ImageContentJitter', ),
    #     ],
    # ),
    # dict(
    #     type='RandomApply',
    #     prob=0.4,
    #     transforms=[
    #         dict(
    #             type='ImgAugWrapper',
    #             args=[dict(cls='AdditiveGaussianNoise', scale=0.1**0.5)]),
    #     ],
    # ),
    # dict(
    #     type='RandomApply',
    #     prob=0.4,
    #     transforms=[
    #         dict(type='ReversePixels', ),
    #     ],
    # ),
    dict(     
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=576,
        width_divisor=8),
    dict(type='PadToWidth', width=576),

    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(     
        type='RescaleToHeight',
        height=32,
        min_width=32,
        max_width=576,
        width_divisor=8),
    dict(type='PadToWidth', width=576),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='ConditionApply',
                true_transforms=[
                    dict(
                        type='ImgAugWrapper',
                        args=[dict(cls='Rot90', k=0, keep_size=False)])
                ],
                condition="results['img_shape'][1]<results['img_shape'][0]"),
            dict(
                type='ConditionApply',
                true_transforms=[
                    dict(
                        type='ImgAugWrapper',
                        args=[dict(cls='Rot90', k=1, keep_size=False)])
                ],
                condition="results['img_shape'][1]<results['img_shape'][0]"),
            dict(
                type='ConditionApply',
                true_transforms=[
                    dict(
                        type='ImgAugWrapper',
                        args=[dict(cls='Rot90', k=3, keep_size=False)])
                ],
                condition="results['img_shape'][1]<results['img_shape'][0]"),
        ], [dict(type='Resize', scale=(256, 64))],
                    [dict(type='LoadOCRAnnotations', with_text=True)],
                    [
                        dict(
                            type='PackTextRecogInputs',
                            meta_keys=('img_path', 'ori_shape', 'img_shape',
                                       'valid_ratio'))
                    ]])
]
