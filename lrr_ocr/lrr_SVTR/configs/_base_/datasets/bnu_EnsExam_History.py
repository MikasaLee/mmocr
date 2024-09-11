bnu_EnsExam_History_textrecog_data_root = '/lirunrui/datasets/mmocr_bnu_EnsExam_History'

bnu_EnsExam_History_textrecog_train = dict(
    type='OCRDataset',
    data_root=bnu_EnsExam_History_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

bnu_EnsExam_History_textrecog_test = dict(
    type='OCRDataset',
    data_root=bnu_EnsExam_History_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
