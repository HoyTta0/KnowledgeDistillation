# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 上午11:15
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : teacher.py
# @Software: PyCharm
"""
import re
import time
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from utils import get_time_dif
import torch.nn.functional as F
from importlib import import_module
from sklearn.model_selection import train_test_split
from models.bert import *
from torch.utils.data import DataLoader, TensorDataset

PAD, CLS = '[PAD]', '[CLS]'


# 预测教师模型输出结果
def teacher_predict(dataset):

    config = Config('data')

    test_data = load_embed(dataset.text, dataset.pred)

    model = Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    model.eval()

    predict_all = np.array([], dtype=int)
    p = []

    with torch.no_grad():
        for a,b,c, labels in test_data:
            outputs = model([a,b,c])
            predic = torch.max(outputs, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
            p.append(outputs)
    # return p
    return predict_all, p


def load_embed(x,y, pad_size=32):
    contents_token_ids = []
    contents_seq_len = []
    contents_mask = []
    contents_labels = []

    re_data = []
    for i in x:
        i_re = ''.join(re.findall(r'[A-Za-z0-9\u4e00-\u9fa5]', i))
        re_data.append(i_re.strip())
    for content,label in zip(re_data,y):
    # content = ''.join(re.findall(r'[A-Za-z0-9\u4e00-\u9fa5]', x)).strip()

        token = BertTokenizer.from_pretrained('./bert_pretrain').tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = BertTokenizer.from_pretrained('./bert_pretrain').convert_tokens_to_ids(token)
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents_labels.append(int(label))
        contents_token_ids.append(token_ids)
        contents_seq_len.append(seq_len)
        contents_mask.append(mask)

    contents_labels = torch.LongTensor(contents_labels)
    contents_token_ids = torch.LongTensor(contents_token_ids)
    contents_seq_len = torch.LongTensor(contents_seq_len)
    contents_mask = torch.LongTensor(contents_mask)

    train_loader = DataLoader(TensorDataset(contents_token_ids,contents_seq_len,contents_mask,contents_labels), batch_size=64)

    return train_loader


def teacher_train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # model.load_state_dict(torch.load('data/saved_dict/xlnet.ckpt'))
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (a,b,c, labels) in enumerate(train_iter):
            # print(total_batch)
            outputs = model([a,b,c])
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # for name, w in model.named_parameters():
            #     if w.requires_grad:
            #         print(name)
            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = teacher_evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    teacher_test(config, model, dev_iter)


def teacher_test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = teacher_evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def teacher_evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for a,b,c, labels in data_iter:
            # print(texts)
            outputs = model([a,b,c])
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # data = pd.DataFrame(columns=('label','pred'))
        # data['label'] = labels_all
        # data['pred'] = predict_all
        # data.to_csv('pred.csv', encoding="utf_8_sig")
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
