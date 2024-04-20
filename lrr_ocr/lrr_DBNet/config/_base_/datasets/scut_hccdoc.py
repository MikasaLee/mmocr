scut_hccdoc_textdet_data_root = '/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset'

scut_hccdoc_textdet_train = dict(
    type='OCRDataset',
    data_root=scut_hccdoc_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

scut_hccdoc_textdet_test = dict(
    type='OCRDataset',
    data_root=scut_hccdoc_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
