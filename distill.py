# -*- coding: utf-8 -*-
"""
# @Time    : 2020/4/29 下午1:42
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : distill.py
# @Software: PyCharm
"""

import time
import pandas as pd
from student import *
from teacher import *
from models.bert import *
from sklearn.metrics import classification_report


if __name__ == '__main__':

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    data = pd.read_csv('train.csv', encoding="utf-8", header=None, usecols=[0, 1, 2])
    data = data.drop(index=[0])
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.dropna()
    data.columns = ['user', 'text', 'pred']
    data = data.apply(pd.to_numeric, errors='ignore')

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    train_teacher = 0
    if train_teacher:
        X_train, X_test, y_train, y_test = \
            train_test_split(data['text'], data['pred'], stratify=data['pred'], test_size=0.2, random_state=1)
        train_data = load_embed(X_train, y_train)
        dev_data = load_embed(X_test, y_test)

        config = Config('data')
        T_model = Model(config)
        teacher_train(config,T_model,train_data,dev_data)

    train_student = 1
    if train_student:
        X_train, X_test, y_train, y_test = \
            train_test_split(data['text'], data['pred'], stratify=data['pred'], test_size=0.2, random_state=1)

        student_train(X_train, X_test, y_train, y_test)

    test = 0
    if test:
        data = pd.read_csv('dev.csv', encoding="utf-8", header=None, usecols=[0, 1, 2])
        data = data.sample(frac=1).reset_index(drop=True)
        data = data.dropna()
        data.columns = ['user', 'text', 'pred']
        data = data.apply(pd.to_numeric, errors='ignore')

        s_predict = student_predict(data.text, data.pred)
        print(classification_report(data.pred, s_predict, target_names=[x.strip() for x in open(
            'data/class_multi1.txt').readlines()], digits=4))
        t_predict, _ = teacher_predict(data.text, data.pred)
        print(classification_report(data.pred, t_predict, target_names=[x.strip() for x in open(
            'data/class_multi1.txt').readlines()], digits=4))

        # data['res']=t_predict
        # data.to_csv('res.csv')




