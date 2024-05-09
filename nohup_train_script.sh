#CUDA_VISIBLE_DEVICES=0,1,2,4,7 PORT=29501 tools/dist_train.sh lrr_ocr/lrr_DBNet/config/dbnet/dbnet_resnet18_fpnc_1200e_icdar2017.py 5 

#CUDA_VISIBLE_DEVICES=0,1,2,5,7 PORT=29501 tools/dist_train.sh lrr_ocr/lrr_CRNN/config/crnn/crnn_mini-vgg_5e_mj.py 5

# CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 PORT=29501 tools/dist_train.sh lrr_ocr/lrr_DBNet/config/dbnet/dbnet_resnet18_fpnc_1200e_ScutHccdoc.py 6
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 PORT=29501 tools/dist_train.sh lrr_ocr/lrr_DBNet/config/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_ScutHccdoc.py 7

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 PORT=29501 tools/dist_train.sh lrr_ocr/lrr_maskrcnn/config/maskrcnn/mask-rcnn_resnet50_fpn_160e_ScutHccdoc.py 6
#CUDA_VISIBLE_DEVICES=0,1 PORT=29503 tools/dist_train.sh lrr_ocr/lrr_SAR/config/sar/sar_resnet31_parallel-decoder_5e_st_handwritting_chinese_recog.py 2 --work-dir work_dirs/sar_resnet31_HWCR_20240427/ --resume 
CUDA_VISIBLE_DEVICES=0,1 PORT=29503 tools/dist_train.sh lrr_ocr/lrr_SAR/config/sar/sar_resnet31_parallel-decoder_5e_st_handwritting_chinese_recog.py 2 --work-dir work_dirs/sar_resnet31_HWCR_withNewSize_20240429/ --resume
