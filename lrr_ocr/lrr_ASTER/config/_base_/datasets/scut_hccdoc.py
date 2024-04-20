scut_hccdoc_textrecog_data_root = '/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset'

scut_hccdoc_textrecog_train = dict(
    type='OCRDataset',
    data_root=scut_hccdoc_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

scut_hccdoc_textrecog_test = dict(
    type='OCRDataset',
    data_root=scut_hccdoc_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
