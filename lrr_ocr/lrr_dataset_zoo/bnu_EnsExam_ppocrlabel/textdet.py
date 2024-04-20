data_root = '/lirunrui/datasets/mmocr_bnu_EnsExam_PPOCRLabel'
cache_path = '/lirunrui/datasets/'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',   # 随便写的不起作用
                save_name='datika_v1',   # 这个写成本来是SCUT_HCCDoc_Dataset的，然后解压的时候操作会变成将`cache_path/save_name`复制到`data_root`中。
                md5=None,
                content=['image','annotation'],
                mapping=[['./*.png', 'textdet_imgs/imgs'],['Label.txt', 'annotations/train_label.txt']]), 
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='train_label.txt',
        img_dir='textdet_imgs/imgs'),

    parser=dict(type='PPOCRLabelTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)

config_generator = dict(type='TextDetConfigGenerator')
delete = ['annotations']