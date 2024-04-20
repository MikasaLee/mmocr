import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import sys
sys.path.append("/lirunrui/mmocr")
from mmocr.apis import MMOCRInferencer
ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
result = ocr('../demo/demo_text_ocr.jpg', show=False, print_result=False,out_dir='./show_pic', save_pred=True, save_vis=True)
print(result)