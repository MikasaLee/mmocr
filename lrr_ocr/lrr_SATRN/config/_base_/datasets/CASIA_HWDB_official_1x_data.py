CASIA_HWDB_official_1x_data_textrecog_data_root = '/lirunrui/datasets/mmocr_CASIA_HWDB_official_1x_dataset'

CASIA_HWDB_official_1x_data_textrecog_train = dict(
    type='OCRDataset',
    data_root=CASIA_HWDB_official_1x_data_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

CASIA_HWDB_official_1x_data_textrecog_test = dict(
    type='OCRDataset',
    data_root=CASIA_HWDB_official_1x_data_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
