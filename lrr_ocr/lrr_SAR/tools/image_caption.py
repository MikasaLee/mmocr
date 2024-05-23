# Copyright (c) MikasaLee. All rights reserved.
import argparse
import types
from typing import Dict, List, Optional, Sequence
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import math
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmocr.structures import TextRecogDataSample
from PIL import Image
from imageio import imread
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
import skimage.transform


 
# # 显示中文
font = FontProperties(fname='/lirunrui/fonts_library-master/simsun.ttc', size=14)
matplotlib.rcParams['agg.path.chunksize'] = 10000
# Reference from mmocr/tools/test.py

def parse_args():
    parser = argparse.ArgumentParser(description='show image caption from SAR')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--output_dir', default='./', help='path to output image')
    parser.add_argument(
        '--work-dir',
        help='The directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--save-preds',
        action='store_true',
        help='Dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# Reference from sar_decoder/ParrallelSARDecoder.forward_test()
def caption_attention_image(
        self,
        feat: torch.Tensor,
        out_enc: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`. # 这个应该是写错了，尺寸是(N,D_m)
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`. # 这个应该是写错了，尺寸是(N,D_m)
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing valid_ratio
                information. Defaults to None.

        Returns:
            Tensor: Character probabilities. of shape
            :math:`(N, self.max_seq_len, C)` where :math:`C` is
            ``num_classes``.
        """
        if data_samples is not None:
            assert len(data_samples) == feat.size(0)

        valid_ratios = None
        if data_samples is not None:
            valid_ratios = [
                data_sample.get('valid_ratio', 1.0)
                for data_sample in data_samples
            ] if self.mask else None

        seq_len = self.max_seq_len

        bsz = feat.size(0)
        start_token = torch.full((bsz, ),
                                 self.start_idx,
                                 device=feat.device,
                                 dtype=torch.long)
        # bsz
        start_token = self.embedding(start_token)
        # bsz * emb_dim
        start_token = start_token.unsqueeze(1).expand(-1, seq_len, -1)
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        decoder_input = torch.cat((out_enc, start_token), dim=1)
        # bsz * (seq_len + 1) * emb_dim

        outputs = []
        atten_weights = []
        for i in range(1, seq_len + 1):
            decoder_output,attn_weight = self.caption_2d_attention(  # 这边改成 caption_2d_attention
                decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            char_output = decoder_output[:, i, :]  # bsz * num_classes
            outputs.append(char_output)
            atten_weights.append(attn_weight[:,i,:,:,:])
            _, max_idx = torch.max(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)  # bsz * emb_dim
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding

        outputs = torch.stack(outputs, 1)  # bsz * seq_len * num_classes

        return outputs,atten_weights

# Reference from sar_decoder/ParrallelSARDecoder._2d_attention,原本的计算注意力的方式在这里，但是这个函数并不会返回每一个词的attention weight。
def caption_2d_attention(self,
                      decoder_input: torch.Tensor,
                      feat: torch.Tensor,
                      holistic_feat: torch.Tensor,
                      valid_ratios: Optional[Sequence[float]] = None
                      ) -> torch.Tensor:
        """2D attention layer.

        Args:
            decoder_input (torch.Tensor): Input of decoder RNN.
            feat (torch.Tensor): Feature map of encoder.
            holistic_feat (torch.Tensor): Feature map of holistic encoder.
            valid_ratios (Sequence[float]): Valid ratios of attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output of 2D attention layer.
        """
        y = self.rnn_decoder(decoder_input)[0]
        # y: bsz * (seq_len + 1) * hidden_size

        attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
        bsz, seq_len, attn_size = attn_query.size()
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)

        attn_key = self.conv3x3_1(feat)
        # bsz * attn_size * h * w
        attn_key = attn_key.unsqueeze(1)
        # bsz * 1 * attn_size * h * w

        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
        # bsz * (seq_len + 1) * attn_size * h * w
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        # bsz * (seq_len + 1) * h * w * attn_size
        attn_weight = self.conv1x1_2(attn_weight)
        # bsz * (seq_len + 1) * h * w * 1
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1

        if valid_ratios is not None:
            # cal mask of attention weight
            attn_mask = torch.zeros_like(attn_weight)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_mask[i, :, :, valid_width:, :] = 1
            attn_weight = attn_weight.masked_fill(attn_mask.bool(),
                                                  float('-inf'))

        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)


        attn_weight = attn_weight.view(bsz, T, h, w,
                                       c).permute(0, 1, 4, 2, 3).contiguous()
        # print("attn_weight.shape:",attn_weight.shape) # attn_weight.shape: torch.Size([1, 31, 1, 16, 144])
        
        attn_feat = torch.sum(
            torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)
        # bsz * (seq_len + 1) * C

        # linear transformation
        if self.pred_concat:
            hf_c = holistic_feat.size(-1)
            holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
            y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)
        # bsz * (seq_len + 1) * num_classes
        y = self.pred_dropout(y)

        return y,attn_weight

# Reference from a-PyTorch-Tutorial-to-Image-Captioning/capiton.py/visualize_att()

def visualize_att(image, text, atten_weights, output_dir):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image (Numpy): after preprocess image,shape: [3,H,W]
    :param text (str): pred_text, '预测的内容'
    :param atten_weights(List(Tensor)): atten weights, len(atten weights) = max_len, atten_weights[0].shape:(1,1,feat_H,feat_W)
    :param output_dir: save the visualize_att path
    """
    print(image.shape)
    
    print(text)
    print('atten_weights info:',len(atten_weights),atten_weights[0].shape) # after decoder,atten_weights info: 30 torch.Size([1, 1, 16, 144])
    print(output_dir)   

    H,W = image.shape[1:]
    image = image * 127 + 127 # [-1,1]
    image = image.astype(int) # [0,255] int

    text = list(text)
    text = ['<start>'] + text + ['<end>']

    plt.figure(figsize=(64, 48),dpi=200)
    
    for t in range(len(text)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(text) / 5.)), 5, t + 1)
        print(text[t])
        plt.text(0, 1, '%s' % (text[t]), color='black', backgroundcolor='white', fontsize=12, fontproperties=font)
        print("image info",image.shape,image.max(),image.min())
        
        plt.imshow(image.transpose(1,2,0))
        current_atten_weight = atten_weights[t].squeeze(0).squeeze(0).detach().numpy()

        atten_weight = skimage.transform.resize(current_atten_weight, [H, W])
        print("atten_weight info",atten_weight.shape,atten_weight.max(),atten_weight.min())
        if t == 0:
            plt.imshow(atten_weight, alpha=0)
        else:
            plt.imshow(atten_weight, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig(output_dir)
    plt.show()
    plt.close()



def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

 
    cfg.load_from = args.checkpoint
    output_dir = None
    output_dir = os.path.join(osp.dirname(args.img),'vis_result_'+os.path.basename(args.img))

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # load model
    checkpoint = torch.load(runner._load_from)

    model = runner.model
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    model = model.to('cpu')
    print('load model successed!')

    # process image
    # 拿出dataloader中的看看
    test_dataloader = runner.test_dataloader
    data = next(iter(test_dataloader))
    input_ = data['inputs']
    data_samples = data['data_samples']
    input_ = input_[0]   # [3,128,2304]
    print(input_.shape,input_.max(),input_.min()) # torch.Size([3, 128, 2304]) tensor(255, dtype=torch.uint8) tensor(0, dtype=torch.uint8)
    # 不在dataloader中做数据预处理，而是在模型中，这点从_base_sar_resnet31_parallel-decoder_chinese.py文件中也能看到,pipeline只负责把尺寸干到(128,2304)，而正则化在model中配置。
    
    # 对一张图片进行处理
    image = imread(args.img)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.concatenate([image, image, image], axis=2)
    image = image.transpose(2, 0, 1)
    image = torch.FloatTensor(image)
    print(image.shape)

    pipeline = runner.test_dataloader.dataset.datasets[0].pipeline
    print(pipeline)
    results = {     # 构造满足pipeline的输入
        'img_path':args.img,
        'instances':[
            {'text':'asdfasdfasf',}   # 瞎写就行，毕竟只是为了看可视化，不需要训练，也就不需要真正的label
        ],
    }
    results = pipeline(results)
    print(results.keys())
    print('after pipeline,image.shape:',results['inputs'].shape) #  torch.Size([3, 128, 2304])

    # 到这里数据以及模型的准备工作就搞定了，可以参考 a-PyTorch-Tutorial-to-Image-Captioning/capiton.py 进行操作了。
    # step1. data_preprocessor
    # 这一步还是要用results走整个流程，而不要 image = results['inputs']这样抽出来走。
    model_data_preprocessor = model.data_preprocessor

    #两个加维度的操作
    results['inputs'] = results['inputs'].unsqueeze(0) # torch.Size([1, 3, 128, 2304])
    results['data_samples'] = [results['data_samples']] # 外面用list包起来。
    
    results = model_data_preprocessor(results)
    print('after preprocessor,image info:',results['inputs'].shape,results['inputs'].max(),results['inputs'].min()) # torch.Size([1, 3, 128, 2304]) tensor(1.0079, device='cuda:0') tensor(-1., device='cuda:0')
   
    # step2. backbone
    # 这一步就是抽出image走
    model_backbone = model.backbone
    inputs = results['inputs']
    feat = model_backbone(inputs)
    print('after backbone,feat info:',feat.shape,feat.max(),feat.min()) # torch.Size([1, 512, 16, 144]) tensor(29.6072, device='cuda:0', grad_fn=<MaxBackward1>) tensor(0., device='cuda:0', grad_fn=<MinBackward1>)
   
    # step3. encoder
    model_encoder = model.encoder
    out_enc = model_encoder(feat)
    print('after encoder,out_enc info:',out_enc.shape,out_enc.max(),out_enc.min()) # torch.Size([1, 512]) tensor(2.1568, device='cuda:0', grad_fn=<MaxBackward1>) tensor(-2.4947, device='cuda:0', grad_fn=<MinBackward1>)
   
    # step4. decoder
    # 模拟decoder.forward_test()函数，然后保存中间变量
    model_decoder = model.decoder
    model_decoder.caption_2d_attention = types.MethodType(caption_2d_attention,model_decoder)  # python 中给类动态添加方法
    model_decoder.caption_attention_image = types.MethodType(caption_attention_image,model_decoder)

    outputs,atten_weights = model_decoder.caption_attention_image(feat=feat,out_enc=out_enc)
    print('after decoder,outputs info:',len(outputs),outputs[0].shape)  # after decoder,outputs info: 1 torch.Size([30, 11380])
    print('after decoder,atten_weights info:',len(atten_weights),atten_weights[0].shape) # after decoder,atten_weights info: 30 torch.Size([1, 1, 16, 144])
    outputs_post = model_decoder.postprocessor(outputs,results['data_samples'])
    # print('after postprocessor,outputs info:',outputs_post) 
    
    visualize_att(results['inputs'].detach().numpy()[0], outputs_post[0].pred_text.item,atten_weights,output_dir)

if __name__ == '__main__':
    main()
