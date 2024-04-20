data_root = '/lirunrui/datasets/mmocr_CASIA_HWDB_chinese_ocr_dataset'
cache_path = '/lirunrui/datasets/handwrite'

# 这个数据集把全部的都当做训练集。
train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',   # 随便写的不起作用
                save_name='chineseocr-data',   # 这个写成本来是SCUT_HCCDoc_Dataset的，然后解压的时候操作会变成将`cache_path/save_name`复制到`data_root`中。
                md5=None,
                content=['image'],
                mapping=[['config', 'textrecog_imgs/imgs']]), 
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip', # 随便写的不起作用
                save_name='chineseocr-data', # 这个写成本来是SCUT_HCCDoc_Dataset的，上面已经复制过了，这里通过is_extract就不复制了。所以这里也是不起作用的
                md5=None,
                is_extract = False, 
                content=['annotation'],
                mapping=[['rec_hand_line_all_label.txt', 'annotations/label.txt']]), #根据官方的说法：对于一个标注文件包含所有图像的标注信息的情况，标注移到到annotations/{split}.*文件中。 如 annotations/train.json。
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='label.txt',
        img_dir='textrecog_imgs/imgs'),

    parser=dict(type='CasiaHwdbChineseOcrTextRecogAnnParser',remove_pre_path='chineseocr-data/config/'),
    packer=dict(type='TextRecogPacker'),
    dumper=dict(type='JsonDumper'),
)

config_generator = dict(type='TextRecogConfigGenerator')