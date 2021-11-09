# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 上午11:15
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : teacher.py
# @Software: PyCharm
"""
import time
import torch
import numpy as np
from models.bert import *
from sklearn import metrics
from utils import get_time_dif


def teacher_predict(model, config, loader):
    model.eval()
    model.load_state_dict(torch.load(config.teacher_save_path))
    t_logits = []
    with torch.no_grad():
        for ids, mask, labels in loader:
            # print(texts)
            ids = ids.to(config.device)
            mask = mask.to(config.device)
            outputs = model(ids, mask)
            t_logits.append(outputs)
    return t_logits


def teacher_train(model, config, train_loader, test_loader):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # model.load_state_dict(torch.load('data/saved_dict/xlnet.ckpt'))
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.teacher_num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.teacher_num_epochs))
        for i, (ids, mask , labels) in enumerate(train_loader):
            # print(total_batch)
            ids = ids.to(config.device)
            mask = mask.to(config.device)
            outputs = model(ids, mask)
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
                dev_acc, dev_loss = teacher_evaluate(model, config, test_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.teacher_save_path)
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
    teacher_test(model, config, test_loader)


def teacher_test(model, config, test_loader):
    # test
    model.load_state_dict(torch.load(config.teacher_save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = teacher_evaluate(model, config, test_loader, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def teacher_evaluate(model, config, test_loader, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for ids, mask , labels in test_loader:
            # print(texts)
            ids = ids.to(config.device)
            mask = mask.to(config.device)
            outputs = model(ids, mask)
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
        return acc, loss_total / len(test_loader), report, confusion
    return acc, loss_total / len(test_loader)
