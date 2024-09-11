bnu_EnsExam_English_textrecog_data_root = '/lirunrui/datasets/mmocr_bnu_EnsExam_English'

bnu_EnsExam_English_textrecog_train = dict(
    type='OCRDataset',
    data_root=bnu_EnsExam_English_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

bnu_EnsExam_English_textrecog_test = dict(
    type='OCRDataset',
    data_root=bnu_EnsExam_English_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
