# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 上午10:53
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : student.py
# @Software: PyCharm
"""
import re
import jieba
import torch
import gensim
import numpy as np
from models.biLSTM import *
from teacher import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report


w2v_model = gensim.models.KeyedVectors.load_word2vec_format('sgns.wiki.word')


def data2frame(x, y):
    data = pd.DataFrame()
    data['text'] = x
    data['pred'] = y
    return data


# 生成句向量
def build_sentence_vector(sentence,w2v_model):

    sen_vec = [0]*300
    count = 0
    for word in sentence:
        try:
            sen_vec += w2v_model[word]
            count += 1
        except KeyError:
            continue
    if count != 0:
        sen_vec /= count
    return sen_vec


# 按一定格式构造学生数据输入
def get_train_data(dataset):

    def load_dataset(x1, y):
        re_data = []
        f_data = []
        l_data = []
        for i in x1:
            i_re = ''.join(re.findall(r'[A-Za-z0-9\u4e00-\u9fa5]', i))
            re_data.append(i_re.strip())
        for content, label in zip(re_data,y):
            token = jieba.cut(content)
            token = [word for word in token]
            token = build_sentence_vector(token, w2v_model)
            f_data.append(token)
            l_data.append(int(label))
        return f_data, l_data

    feature_data, label_data = load_dataset(dataset.text, dataset.pred)
    feature_data, label_data = np.array(feature_data), np.array(label_data)
    train_x = [feature_data[i:i + 1] for i in range(len(dataset))]
    train_y = [label_data[i:i + 1] for i in range(len(dataset))]
    train_x, train_y = np.array(train_x), np.array(train_y)
    train_X, train_Y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=64)

    return train_loader


# 损失函数
def get_loss(t_logits, s_logits, label, a, T):
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    loss = a * loss1(s_logits, label) #+ T * loss2(t_logits, s_logits)
    # print(loss1(s_logits, label),loss2(t_logits, s_logits))
    return loss


# 预测学生模型输出结果
def student_predict(dataset):
    model = biLSTM()
    data = get_train_data(dataset)
    model.load_state_dict(torch.load('data/saved_dict/lstm.ckpt'))
    model.eval()
    predict_all = []
    hidden_predict = None
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    with torch.no_grad():
        for texts, labels in data:
            pred_X, hidden_predict = model(texts, hidden_predict)
            hidden_predict = None
            cur_pred = torch.squeeze(pred_X, dim=1)
            predic = torch.max(cur_pred, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)

    return predict_all


def student_train(dataset):
    X_train, X_test, y_train, y_test = \
        train_test_split(dataset['text'], dataset['pred'], stratify=dataset['pred'], test_size=0.2, random_state=1)
    train_student = data2frame(X_train, y_train)
    test_student = data2frame(X_test, y_test)
    _, t_logits = teacher_predict(train_student)
    _, t_test = teacher_predict(test_student)
    train_loader = get_train_data(train_student)
    student = biLSTM()
    total_params = sum(p.numel() for p in student.parameters())
    print(f'{total_params:,} total parameters.')
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    total_batch = 0
    total_epoch = 50
    tra_best_loss = float('inf')
    dev_best_loss = float('inf')
    student.train()
    start_time = time.time()
    for epoch in range(total_epoch):
        hidden_train = None
        print('Epoch [{}/{}]'.format(epoch + 1, total_epoch))
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            s_logits, _ = student(x, hidden_train)
            hidden_train = None
            label = y.squeeze(1).long()
            loss = get_loss(t_logits[i], s_logits.squeeze(1), label, 1, 3)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                cur_pred = torch.squeeze(s_logits, dim=1)
                train_acc = metrics.accuracy_score(y.squeeze(1).long(), torch.max(cur_pred, 1)[1].cpu().numpy())
                _, dev_loss, dev_acc = student_evaluate(test_student, student, t_test)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(student.state_dict(), 'data/saved_dict/lstm.ckpt')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                student.train()
            total_batch += 1

    student_test(test_student)


def student_evaluate(dataset, model, t_logits):
    data = get_train_data(dataset)
    model.eval()
    predict_all = []
    labels_all = []
    hidden_predict = None
    loss_total = 0
    with torch.no_grad():
        for i, (texts, labels) in enumerate(data):
            pred_X, hidden_predict = model(texts, hidden_predict)
            hidden_predict = None
            cur_pred = torch.squeeze(pred_X, dim=1)
            # loss = F.cross_entropy(cur_pred.squeeze(1), labels.squeeze(1).long())
            loss = get_loss(t_logits[i], pred_X.squeeze(1), labels.squeeze(1).long(), 1, 3)
            loss_total += loss
            predic = torch.max(cur_pred, 1)[1].cpu().numpy()
            labels = labels.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return predict_all, loss_total/len(data), acc


def student_test(dataset):
    # test
    y= student_predict(dataset)
    print(classification_report(dataset.pred, y, target_names=[x.strip() for x in open(
            'data/class_multi.txt').readlines()], digits=4))
