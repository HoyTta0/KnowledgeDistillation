# -*- coding: utf-8 -*-
"""
# @Time    : 2021/11/7 8:30 下午
# @Author  : HOY
# @Email   : 893422529@qq.com
# @File    : config.py
"""
import torch
from transformers import BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self):
        self.class_list = [x.strip() for x in open(
            'data/class_multi1.txt').readlines()]                                # 类别名单
        self.train_path = 'data/train.json'
        self.test_path = 'data/test.json'
        self.teacher_save_path = 'saved_dict/teacher.ckpt'        # 模型训练结果
        self.student_save_path = 'saved_dict/student.ckpt'        # 模型训练结果

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.train_teacher = 0
        self.train_student = 1
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.teacher_num_epochs = 3                                          # epoch数
        self.student_num_epochs = 3                                          # epoch数

        self.batch_size = 64                                      # 128mini-batch大小
        self.pad_size = 32                                           # 每句话处理成的长度(短填长切)
        self.learning_rate =  5e-4                                    # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768