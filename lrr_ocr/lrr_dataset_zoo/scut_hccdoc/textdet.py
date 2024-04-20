data_root = '/lirunrui/datasets/mmocr_SCUT_HCCDoc_Dataset'
cache_path = '/lirunrui/datasets/'

train_preparer = dict(
    obtainer=dict(
        type='NaiveDataObtainer',
        cache_path=cache_path,
        files=[
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',   # 随便写的不起作用
                save_name='SCUT_HCCDoc_Dataset',   # 这个写成本来是SCUT_HCCDoc_Dataset的，然后解压的时候操作会变成将`cache_path/save_name`复制到`data_root`中。
                md5=None,
                content=['image'],
                mapping=[['image', 'textdet_imgs/imgs']]), # 对于一些其他的特殊情况，比如所有训练、测试、验证的图像都在一个文件夹下，可以将图像移动到自己设定的文件夹下，比如 {taskname}_imgs/imgs/，同时要在后续的 gatherer 模块中指定图像的存储位置。
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip', # 随便写的不起作用
                save_name='SCUT_HCCDoc_Dataset', # 这个写成本来是SCUT_HCCDoc_Dataset的，上面已经复制过了，这里通过is_extract就不复制了。所以这里也是不起作用的
                md5=None,
                is_extract = False,
                content=['annotation'],
                mapping=[['hccdoc_train.json', 'annotations/train.json']]), #根据官方的说法：对于一个标注文件包含所有图像的标注信息的情况，标注移到到annotations/{split}.*文件中。 如 annotations/train.json。
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='train.json',
        img_dir='textdet_imgs/imgs'),

    parser=dict(type='SCUTHCCDocTextDetAnnParser'),
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
                save_name='SCUT_HCCDoc_Dataset',   
                md5=None,
                is_extract = False,   # 不需要复制了
                content=['image']),   # 下一行没有mapping
            dict(
                url='https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip', # 随便写的不起作用
                save_name='SCUT_HCCDoc_Dataset', # 这个写成本来是SCUT_HCCDoc_Dataset的，上面已经复制过了，这里通过is_extract就不复制了。所以这里也是不起作用的
                md5=None,
                is_extract = False,
                content=['annotation'],
                mapping=[['hccdoc_test.json', 'annotations/test.json']]), 
        ]),
    gatherer=dict(
        type='MonoGatherer',
        ann_name='test.json',
        img_dir='textdet_imgs/imgs'),

    parser=dict(type='SCUTHCCDocTextDetAnnParser'),
    packer=dict(type='TextDetPacker'),
    dumper=dict(type='JsonDumper'),
)



config_generator = dict(type='TextDetConfigGenerator')
delete = ['annotations']
