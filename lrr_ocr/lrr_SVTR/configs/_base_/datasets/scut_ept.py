scut_ept_textrecog_data_root = '/lirunrui/datasets/mmocr_SCUT_EPT_Dataset'

scut_ept_textrecog_train = dict(
    type='OCRDataset',
    data_root=scut_ept_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

scut_ept_textrecog_test = dict(
    type='OCRDataset',
    data_root=scut_ept_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
