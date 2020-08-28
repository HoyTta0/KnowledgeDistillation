# -*- coding: utf-8 -*-
"""
# @Time    : 2020/7/16 下午2:06
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : xlnet.py
# @Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel


class Config():

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'xlnet'
        self.train_path = 'train.csv'                                # 训练集
        self.class_list = [x.strip() for x in open(
            dataset + '/class_multi1.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                          # epoch数
        self.batch_size = 64                                      # 128mini-batch大小
        self.pad_size = 32                                           # 每句话处理成的长度(短填长切)
        self.learning_rate =  5e-5                                    # 学习率
        self.xlnet_path = './xlnet_pretrain'
        self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(config.xlnet_path, num_labels=config.num_classes)
        for param in list(self.xlnet.parameters())[:-5]:
            param.requires_grad = False
        self.fc = nn.Linear(config.hidden_size, 192)
        self.fc1 = nn.Linear(192, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        logits = self.xlnet(input_ids=context, attention_mask=mask)
        logits = logits[0]
        out = logits[:, -1]
        out = self.fc(out)
        out = F.relu(out)
        out = self.fc1(out)
        return out
