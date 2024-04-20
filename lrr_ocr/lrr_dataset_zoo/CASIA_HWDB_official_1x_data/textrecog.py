data_root = '/lirunrui/datasets/mmocr_CASIA_HWDB_official_1x_dataset'
cache_path = '/lirunrui/datasets/'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',   # 随便写的不起作用
                save_name='handwritten-chinese-character-hanzi-datasets',   # 这个写成本来是SCUT_HCCDoc_Dataset的，然后解压的时候操作会变成将`cache_path/save_name`复制到`data_root`中。
                md5=None,
                content=['image','annotation'],
                mapping=[['CASIA-HWDB_Train/Train', 'textrecog_imgs/train'],['train_label.txt', 'annotations/train_label.txt']]),   # 一定要注意train_label.txt写对，别tm写成了train_labels.txt，md这个问题搞了半天。
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='train_label.txt',
        img_dir='textrecog_imgs/train'),

    parser=dict(type='CasiaHwdbChineseOcrTextRecogAnnParser',remove_pre_path='./CASIA-HWDB_Train/Train/'), # 直接用之前的parser就行，改了remove_pre_path就可以。
    packer=dict(type='TextRecogPacker'),
    dumper=dict(type='JsonDumper'),
)

test_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',   # 随便写的不起作用
                save_name='handwritten-chinese-character-hanzi-datasets',   # 这个写成本来是SCUT_HCCDoc_Dataset的，然后解压的时候操作会变成将`cache_path/save_name`复制到`data_root`中。
                md5=None,
                is_extract = False, 
                content=['image','annotation'],
                mapping=[['CASIA-HWDB_Test/Test', 'textrecog_imgs/test'],['test_label.txt', 'annotations/test_label.txt']]), 
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='test_label.txt',
        img_dir='textrecog_imgs/test'),

    parser=dict(type='CasiaHwdbChineseOcrTextRecogAnnParser',remove_pre_path='./CASIA-HWDB_Test/Test/'),
    packer=dict(type='TextRecogPacker'),
    dumper=dict(type='JsonDumper'),
)

config_generator = dict(type='TextRecogConfigGenerator')