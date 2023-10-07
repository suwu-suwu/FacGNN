from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from module.PositionEmbedding import get_sinusoid_encoding_table

WORD_PAD = "[PAD]"


class sentEncoder(nn.Module):
    def __init__(self, hps, embed):
        """
        :param hps: 
                word_emb_dim: word embedding dimension
                sent_max_len: max token number in the sentence
                word_embedding: bool, use word embedding or not
                embed_train: bool, whether to train word embedding
                cuda: bool, use cuda or not
        """
        super(sentEncoder, self).__init__()

        self._hps = hps
        self.sent_max_len = hps.sent_max_len
        embed_size = hps.word_emb_dim

        input_channels = 1
        out_channels = 50
        min_kernel_size = 2
        max_kernel_size = 7
        width = embed_size

        # word embedding
        self.embed = embed

        # position embedding
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.sent_max_len + 1, embed_size, padding_idx=0), freeze=True)

        # cnn
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, out_channels, kernel_size=(height, width),dilation=1) for height in
                                    range(min_kernel_size, max_kernel_size + 1)])
        

        for conv in self.convs:
            init_weight_value = 6.0
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))

    def forward(self, input):
        # input: a batch of Example object [s_nodes, seq_len]
        input_sent_len = ((input != 0).sum(dim=1)).int()  # [s_nodes, 1]
        enc_embed_input = self.embed(input.cuda())  # [s_nodes, L, D]
        
        sent_pos_list = []
        for sentlen in input_sent_len:
            sent_pos = list(range(1, min(self.sent_max_len, sentlen) + 1))
            sent_pos.extend([0] * int(self.sent_max_len - sentlen))
            sent_pos_list.append(sent_pos)
        input_pos = torch.Tensor(sent_pos_list).long()

        if self._hps.cuda:
            input_pos = input_pos.cuda()
        enc_pos_embed_input = self.position_embedding(input_pos.long())  # [s_nodes, D]
        enc_conv_input = enc_embed_input + enc_pos_embed_input
        enc_conv_input = enc_conv_input.unsqueeze(1)  # [s_nodes, 1, L, D]

        enc_conv_output = [F.relu(conv(enc_conv_input).squeeze(3)) for conv in self.convs]  # kernel_sizes * [s_nodes, Co=50, W]
        
        enc_maxpool_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in enc_conv_output]  # kernel_sizes * [s_nodes, Co=50]
        sent_embedding = torch.cat(enc_maxpool_output, 1)  # [s_nodes, 50 * 6]
        return sent_embedding ,  enc_conv_input# [s_nodes, 300]
    

class ASPP(nn.Module):
    def __init__(self, in_channel, depth,hps, embed):
        super(ASPP,self).__init__()
        self._hps = hps
        self.sent_max_len = hps.sent_max_len
        embed_size = hps.word_emb_dim
        self.embed = embed
        # position embedding
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.sent_max_len + 1, embed_size, padding_idx=0), freeze=True)
        
        
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        
        # 不同空洞率的卷积
        self.convs = nn.ModuleList( [nn.Conv2d(in_channel, depth, 1, 1),nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)])
        
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        
 
    def forward(self, x):
        input_sent_len = ((x != 0).sum(dim=1)).int()  # [s_nodes, 1]
        enc_embed_input = self.embed(x.cuda())  # [s_nodes, L, D]
        print(enc_embed_input.shape,"dweqdqdq")
        
        sent_pos_list = []
        for sentlen in input_sent_len:
            sent_pos = list(range(1, min(self.sent_max_len, sentlen) + 1))
            sent_pos.extend([0] * int(self.sent_max_len - sentlen))
            sent_pos_list.append(sent_pos)
        input_pos = torch.Tensor(sent_pos_list).long()
        if self._hps.cuda:
            input_pos = input_pos.cuda()
        enc_pos_embed_input = self.position_embedding(input_pos.long())  # [s_nodes, D]
        
        enc_conv_input = enc_embed_input + enc_pos_embed_input 
        x = enc_conv_input.unsqueeze(1)# [s_nodes, 1, L, D]
        
        size = x.shape[2:]
     # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
     # 不同空洞率的卷积
        
        ii=0
        for conv in self.convs:
            ii=ii+1
            if ii==1:
                atrous_block1 = conv(x)
            if ii==2:
                atrous_block6 = conv(x)
            if ii==3:
                atrous_block12 = conv(x)
            if ii==4:
                atrous_block18 = conv(x)
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        
        return x

class SimAM(torch.nn.Module):
    def __init__(self, channels = None,out_channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def forward(self, x):
 
        b, c, h, w = x.size()
        
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)  