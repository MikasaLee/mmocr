# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import json
from mmocr.datasets.preparers.parsers.base import BaseParser
from mmocr.registry import DATA_PARSERS


@DATA_PARSERS.register_module()
class CasiaHwdbChineseOcrTextRecogAnnParser(BaseParser):
    def __init__(self,
                 remove_pre_path: str = None,
                 **kwargs) -> None:

        self.remove_pre_path = remove_pre_path
        super().__init__(**kwargs)

    # 参考: /lirunrui/mmocr/mmocr/datasets/preparers/parsers/icdar_txt_parser.py 的 ICDARTxtTextRecogAnnParser类
    def parse_files(self, img_dir: str, ann_path: str) -> List:
        """Parse single annotation."""
        assert isinstance(ann_path, str)
        samples = list()
        for anno in self.loader(
                file_path=ann_path,
                format='img\ttext',
                encoding='utf-8',
                separator='\t'):

            text = anno['text'].strip()
            if self.remove_pre_path is not None:
                img_name = anno['img'].replace(self.remove_pre_path,'')  #删掉前缀，没必要了
            else:
                img_name = anno['img']
            samples.append((osp.join(img_dir, img_name), text))
        return samples
