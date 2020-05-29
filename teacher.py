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


PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(config, x, y, pad_size=32):
    contents = []
    re_data = []
    for i in x:
        i_re = ''.join(re.findall(r'[A-Za-z0-9\u4e00-\u9fa5]', i))
        re_data.append(i_re.strip())
    for content, label in zip(re_data, y):
        token = config.tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, int(label), seq_len, mask))
    return contents


# 按一定格式构造教师数据输入
def build_predict_dataset(config, dataset):

    data = load_dataset(config, dataset.text, dataset.pred, config.pad_size)
    return data


def build_train_dataset(config):
    data = pd.read_csv(config.train_path, encoding="utf-8", header=None)
    data = data.drop(index=[0])
    data.columns = ['text', 'pred']
    X_train, X_test, y_train, y_test = \
        train_test_split(data['text'], data['pred'], stratify=data['pred'], test_size=0.2, random_state=1)
    train = load_dataset(config, X_train, y_train, config.pad_size)
    dev = load_dataset(config, X_test, y_test, config.pad_size)
    test = load_dataset(config, X_test, y_test, config.pad_size)
    # df = pd.DataFrame()
    # df['text'] = X_test
    # df.to_csv('text.csv', encoding="utf_8_sig",columns=['text'])
    return train, dev, test


class DatasetIterater(object):

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size if len(batches) // batch_size!=0 else 1
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):

    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


# 预测教师模型输出结果
def teacher_predict(dataset):

    model_name = 'bert'
    x = import_module('models.' + model_name)
    config = x.Config('data')

    test_data = build_predict_dataset(config, dataset)
    test_iter = build_iterator(test_data, config)

    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    model.eval()

    predict_all = np.array([], dtype=int)
    p = []

    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts)
            predic = torch.max(outputs, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
            p.append(outputs)
    return p
    # return predict_all


def teacher_train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    tra_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            # print(total_batch)
            outputs = model(trains)
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
    teacher_test(config, model, test_iter)


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
        for texts, labels in data_iter:
            # print(texts)
            outputs = model(texts)
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
