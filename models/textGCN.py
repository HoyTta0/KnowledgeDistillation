# -*- coding: utf-8 -*-
"""
# @Time    : 2020/7/9 下午3:36
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : textGCN.py
# @Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:

    def __init__(self, dataset, l):
        # 预处理参数
        self.window_size = 20  # 20

        # 训练参数
        self.learning_rate = 0.05  # 0.02
        self.max_epoch = 200  # 200
        self.stop_epoch = 10  # 10


        # 模型参数
        self.dropout = 0.5  # 0.5
        self.embedding_dim = 200  # 200
        self.num_nodes = l+len(dataset)
        self.num_document = len(dataset)
        self.num_classes = 2


class textGCN(nn.Module):
    def __init__(self, A, config):
        super(textGCN, self).__init__()
        self.num_nodes = config.num_nodes
        self.num_document = config.num_document
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes

        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float), requires_grad=False)
        self.W0 = nn.Linear(self.num_nodes, self.embedding_dim, bias=True)
        self.W1 = nn.Linear(self.embedding_dim, self.num_classes, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.W0(self.A.mm(x))
        x = F.relu(x, inplace=True)
        x = self.W1(self.A.mm(x))
        x = self.dropout(x)
        x = F.softmax(x[: self.num_document], dim=1)
        return x


