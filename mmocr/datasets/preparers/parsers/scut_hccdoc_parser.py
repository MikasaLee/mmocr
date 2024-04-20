# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import json
from mmocr.datasets.preparers.parsers.base import BaseParser
from mmocr.registry import DATA_PARSERS


@DATA_PARSERS.register_module()
class SCUTHCCDocTextDetAnnParser(BaseParser):

    def loader(self, file_path: str) -> str:
        """Load the annotation of the SCUT-HCCDoc dataset.

        Args:
            file_path (str): Path to the json file

        Retyrb:
            str: Complete annotation of the json file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        annotations_data = data['annotations']

        #  contains the text filled by human.
        for data_type in ['HCCDoc-WS', 'HCCDoc-WSF', 'HCCDoc-WT', 'HCCDoc-SN', 'HCCDoc-EP']:
            for anno_data in annotations_data[data_type]:
                img_file_path = anno_data['file_path']
                image_id = anno_data['image_id']
                h,w = anno_data['height'],anno_data['height']
                gts = anno_data['gt']

                yield img_file_path,image_id,h,w,gts

    def parse_files(self, img_dir: str, ann_path: str) -> List:
        """Parse single annotation."""
        samples = list()
        for img_file_path, image_id, h, w, gts in self.loader(ann_path):
            instances = list()
            for gt in gts:
                instances.append(
                    dict(
                        poly=gt['point'],
                        text=gt['text'],
                        ignore=False))

            samples.append((osp.join(img_dir,
                                     img_file_path), instances))
        return samples
