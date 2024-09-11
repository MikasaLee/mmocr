data_root = '/lirunrui/datasets/mmocr_bnu_EnsExam_Politics'
cache_path = '/a800data1/lirunrui/origin_datasets/'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',   # 随便写的不起作用
                save_name='bnu_EnsExam_Politics',
                md5=None,
                content=['image','annotation'],
                mapping=[['./train/*.png', 'textdet_imgs/train'],['./train/Label.txt', 'annotations/train_label.txt']]), 
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='train_label.txt',
        img_dir='textdet_imgs/train'),

    parser=dict(type='PPOCRLabelTextDetAnnParser',
        is_replace_UNRECOG=True,
        is_replace_IMAGE=True,
        is_remove_extra_space= True, 
        is_Chinese_to_convert_English_for_symbols=False,
        is_replace_char_without_dict=True,
        dict_path='/lirunrui/mmocr/dicts/chinese_english_digits_2.txt'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',   # 随便写的不起作用
                save_name='bnu_EnsExam_Politics',   # 这个写成本来是SCUT_HCCDoc_Dataset的，然后解压的时候操作会变成将`cache_path/save_name`复制到`data_root`中。
                md5=None,
                is_extract = False, 
                content=['image','annotation'],
                mapping=[['./test/*.png', 'textdet_imgs/test'],['./test/Label.txt', 'annotations/test_label.txt']]), 
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='test_label.txt',
        img_dir='textdet_imgs/test'),

    parser=dict(type='PPOCRLabelTextDetAnnParser',
        is_replace_UNRECOG=True,
        is_replace_IMAGE=True,
        is_remove_extra_space= True, 
        is_Chinese_to_convert_English_for_symbols=False,
        is_replace_char_without_dict=True,
        dict_path='/lirunrui/mmocr/dicts/chinese_english_digits_2.txt'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)
config_generator = dict(type='TextDetConfigGenerator')
delete = ['annotations']