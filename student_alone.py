# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/28 上午11:09
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : student_alone.py
# @Software: PyCharm
"""
from student import *
from models.biLSTM import *


def student_train_alone(dataset):
    print(1)
    train_loader = get_train_data(dataset)
    student = biLSTM()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    tra_best_loss = float('inf')
    student.train()
    losss = nn.CrossEntropyLoss()
    for epoch in range(200):
        hidden_train = None
        print('Epoch [{}/{}]'.format(epoch + 1, 200))
        for x,y in train_loader:
            optimizer.zero_grad()
            pred_X, _ = student(x, hidden_train)
            hidden_train = None
            # print(pred_X.squeeze(1).dtype,y.squeeze(1).dtype)
            loss = losss(pred_X.squeeze(1),y.squeeze(1).long())
            loss.backward()
            optimizer.step()
            if loss.item() < tra_best_loss:
                tra_best_loss = loss.item()
                torch.save(student.state_dict(), 'data/saved_dict/lstm_student.ckpt')
                print(loss.item())


def predict_alone(dataset):
    model = biLSTM()
    # model = StudentNet()
    valid_data = get_train_data(dataset)
    model.load_state_dict(torch.load('data/saved_dict/lstm_student.ckpt'))
    model.eval()
    predict_all = []
    hidden_predict = None
    result = torch.Tensor()
    with torch.no_grad():
       for texts, labels in valid_data:
           pred_X, hidden_predict = model(texts, hidden_predict)
           # pred_X = model(texts)
           hidden_predict = None
           cur_pred = torch.squeeze(pred_X, dim=1)
           result = torch.cat((result, cur_pred), dim=0)
           predic = torch.max(cur_pred, 1)[1].cpu().numpy()
           predict_all = np.append(predict_all, predic)
    # print(result.detach().numpy())
    return predict_all