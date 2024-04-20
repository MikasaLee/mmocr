# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import json
from mmocr.datasets.preparers.parsers.base import BaseParser
from mmocr.registry import DATA_PARSERS


@DATA_PARSERS.register_module()
class CasiaHwdbOfficial2xTextRecogAnnParser(BaseParser):
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
            img_name = anno['img'].replace('handwrite/hwdb_ic13/handwriting_hwdb_train_images/','')  #删掉前缀，没必要了

            samples.append((osp.join(img_dir, img_name), text))
        return samples
