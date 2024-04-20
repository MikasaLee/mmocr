bnu_EnsExam_ppocrlabel_textdet_data_root = '/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel'

bnu_EnsExam_ppocrlabel_textdet_train = dict(
    type='OCRDataset',
    data_root=bnu_EnsExam_ppocrlabel_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

bnu_EnsExam_ppocrlabel_textdet_test = dict(
    type='OCRDataset',
    data_root=bnu_EnsExam_ppocrlabel_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
