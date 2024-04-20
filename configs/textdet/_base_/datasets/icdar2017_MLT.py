icdar2017_MLT_textdet_data_root = '/lirunrui/datasets/icdar2017_MLT/ICDAR2017_MLT'

icdar2017_MLT_textdet_train = dict(
    type='OCRDataset',
    data_root=icdar2017_MLT_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

icdar2017_MLT_textdet_test = dict(
    type='OCRDataset',
    data_root=icdar2017_MLT_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
