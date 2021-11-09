# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 上午10:53
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : student.py
# @Software: PyCharm
"""
from teacher import *


# 生成句向量
def build_sentence_vector(sentence,w2v):

    sen_vec = [0]*300
    count = 0
    for word in sentence:
        try:
            sen_vec += w2v[word]
            count += 1
        except KeyError:
            continue
    if count != 0:
        sen_vec /= count
    return sen_vec


# 损失函数
def get_loss(t_logits, s_logits, label, a, T):
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    loss = a * loss1(s_logits, label) + T * loss2(t_logits, s_logits)
    # print(loss1(s_logits, label),loss2(t_logits, s_logits))
    return loss


def student_train(T_model, S_model, config, train_loader, test_loader):
    t_train_logits = teacher_predict(T_model, config, train_loader)
    t_test_logits = teacher_predict(T_model, config, test_loader)
    total_params = sum(p.numel() for p in S_model.parameters())
    print(f'{total_params:,} total parameters.')
    optimizer = torch.optim.SGD(S_model.parameters(), lr=0.05)
    total_batch = 0
    tra_best_loss = float('inf')
    dev_best_loss = float('inf')
    S_model.train()
    start_time = time.time()
    for epoch in range(config.student_num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.student_num_epochs))
        for i, (texts, _, label) in enumerate(train_loader):
            texts = texts.to(config.device)
            label = label.to(config.device)
            optimizer.zero_grad()
            s_logits = S_model(texts)
            loss = get_loss(t_train_logits[i], s_logits, label.long(), 1, 2)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                cur_pred = torch.squeeze(s_logits, dim=1)
                train_acc = metrics.accuracy_score(label.long(), torch.max(cur_pred, 1)[1].cpu().numpy())
                dev_loss, dev_acc = student_evaluate(S_model, config, t_test_logits, test_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(S_model.state_dict(), config.student_save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                S_model.train()
            total_batch += 1
    student_evaluate(S_model, config, t_test_logits, test_loader)


def student_evaluate(S_model, config, t_logits, test_loader):
    S_model.eval()
    predict_all = []
    labels_all = []
    loss_total = 0
    with torch.no_grad():
        for i, (texts, _, label) in enumerate(test_loader):
            texts = texts.to(config.device)
            label = label.to(config.device)
            s_logits = S_model(texts)
            loss = get_loss(t_logits[i], s_logits, label.long(), 1, 2)
            loss_total += loss

            cur_pred = torch.squeeze(s_logits, dim=1)
            predic = torch.max(cur_pred, 1)[1].cpu().numpy()
            label = label.data.cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return loss_total/len(test_loader), acc



