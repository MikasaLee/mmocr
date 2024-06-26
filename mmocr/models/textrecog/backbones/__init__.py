# Copyright (c) OpenMMLab. All rights reserved.
from .mini_vgg import MiniVGG
from .mobilenet_v2 import MobileNetV2
from .nrtr_modality_transformer import NRTRModalityTransform
from .resnet import ResNet
from .resnet31_ocr import ResNet31OCR
from .resnet_abi import ResNetABI
from .shallow_cnn import ShallowCNN
from .scut_hccdoc_backbone import ScutHccdocBackbone
__all__ = [
    'ResNet31OCR', 'MiniVGG', 'NRTRModalityTransform', 'ShallowCNN',
    'ResNetABI', 'ResNet', 'MobileNetV2','ScutHccdocBackbone'
]
