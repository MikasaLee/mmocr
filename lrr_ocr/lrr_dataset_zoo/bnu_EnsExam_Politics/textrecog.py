_base_ = ['textdet.py']

_base_.train_preparer.packer.type = 'TextRecogCropPacker'
_base_.train_preparer.packer.is_crop_or_whiten = True  # 不用裁剪，用涂白

_base_.test_preparer.packer.type = 'TextRecogCropPacker'
_base_.test_preparer.packer.is_crop_or_whiten = True   # 不用裁剪，用涂白


config_generator = dict(type='TextRecogConfigGenerator')
