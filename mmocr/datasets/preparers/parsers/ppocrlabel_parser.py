# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import json
from mmocr.datasets.preparers.parsers.base import BaseParser
from mmocr.registry import DATA_PARSERS


@DATA_PARSERS.register_module()
class PPOCRLabelTextDetAnnParser(BaseParser):

    def parse_files(self, img_dir: str, ann_path: str) -> List:
        """Parse single annotation."""
        assert isinstance(ann_path, str)
        samples = list()
        for anno in self.loader(
                file_path=ann_path,
                format='img\ttext',
                encoding='utf-8',
                separator='\t'):

            img_name = anno['img'].split('/')[1]   # 只要 xx.png
            label = anno['text'].strip()


            json_datas = json.loads(label)
            instances = list()
            for json_data in json_datas:
                box = json_data['points']  # PPOCRLabel 读出来是二维的
                box = [i for p in box for i in p]   # 展平成一维的
                
                text = json_data['transcription']
                instances.append(
                    dict(
                        poly=box,
                        text=text,
                        ignore=False))

            samples.append((osp.join(img_dir, img_name), instances))
        return samples