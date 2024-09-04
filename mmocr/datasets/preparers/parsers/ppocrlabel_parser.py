# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re
from typing import List

import json
from mmocr.datasets.preparers.parsers.base import BaseParser
from mmocr.registry import DATA_PARSERS


@DATA_PARSERS.register_module()
class PPOCRLabelTextDetAnnParser(BaseParser):
    def __init__(self,
                 is_replace_UNRECOG = False,   # 是否用#替换掉<UNRECOG>
                 is_replace_IMAGE = False,   # 是否用#替换掉<IMAGE>
                 is_remove_extra_space= False,   # 是否去掉多余的空格
                 is_Chinese_to_convert_English_for_symbols = False,   # 是否将中文符号换成英文
                 is_replace_char_without_dict = False,   # 是否将所有没有出现在字典中的字符全部替换为`#`
                 dict_path = None, # 如果is_replace_char_without_dict为True,需要指定字典路径
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.is_replace_UNRECOG = is_replace_UNRECOG
        self.is_replace_IMAGE = is_replace_IMAGE
        self.is_remove_extra_space = is_remove_extra_space
        self.is_Chinese_to_convert_English_for_symbols = is_Chinese_to_convert_English_for_symbols
        self.is_replace_char_without_dict = is_replace_char_without_dict
        self.is_replace_char_without_dict = is_replace_char_without_dict
        self.dict_path = dict_path
        if self.is_replace_char_without_dict == True and self.dict_path == None:
            raise Error('is_replace_char_without_dict is True BUT dict_path is None!')
        if self.dict_path:
            self.dict = ' '    # strip()会把空格也给去掉，所以一开始就加上空格
            with open(self.dict_path, "r",encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    self.dict += line
            print("字典内容：",self.dict)


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

                ## add process
                if self.is_replace_UNRECOG:
                    text = self.replace_UNRECOG(text)

                if self.is_replace_IMAGE:
                    text = self.replace_IMAGE(text)

                if self.is_remove_extra_space:
                    text = self.remove_extra_space(text)

                if self.is_Chinese_to_convert_English_for_symbols:
                    text = self.Chinese_to_convert_English_for_symbols(text)

                if self.is_replace_char_without_dict:
                    text = self.replace_char_without_dict(text)
                    
                instances.append(
                    dict(
                        poly=box,
                        text=text,
                        ignore=False))

            samples.append((osp.join(img_dir, img_name), instances))
        return samples

    def replace_UNRECOG(self,text):
        return text.replace('<UNRECOG>','#')
    def replace_IMAGE(self,text):
        return text.replace('<IMAGE>','#')
    def remove_extra_space(self,text):
        text = text.strip()
        result = re.sub(r'\s+', ' ', text)
        return result
    def Chinese_to_convert_English_for_symbols(self,text):
        zh_to_en_map = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '：': ':',
            '；': ';',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '—': '-',
            '…': '...'
        }
        for chinese, english in zh_to_en_map.items():
            text = text.replace(chinese, english)
        return text

    def replace_char_without_dict(self,text):
        new_text = ''
        for idx,char in enumerate(text):
            if char not in self.dict: new_text += '#'
            else: new_text += char
        return new_text