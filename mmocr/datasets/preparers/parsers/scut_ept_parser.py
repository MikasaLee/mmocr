# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import json
from mmocr.datasets.preparers.parsers.base import BaseParser
from mmocr.registry import DATA_PARSERS

@DATA_PARSERS.register_module()
class SCUTEPTTextRecogAnnParser(BaseParser):
    def parse_files(self, img_dir: str, ann_path: str) -> List:
        """Parse single annotation."""
        assert isinstance(ann_path, str)
        samples = list()

        idx = 0
        for anno in self.loader(
            file_path=ann_path,
            format='text',
            encoding='utf-8'):

            text = anno['text'].strip()
            img_name = str(idx).zfill(6) + '.jpg'
            # print("idx:{}\t img_name:{}".format(idx,img_name))  # check
            idx = idx + 1
            samples.append((osp.join(img_dir, img_name), text))
        return samples
