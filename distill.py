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
from sklearn.metrics import classification_report


if __name__ == '__main__':
    train_teacher = 0
    if train_teacher:

        x = import_module('models.bert')
        config = x.Config('data')
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        start_time = time.time()
        print("Loading data...")
        train_data, dev_data, test_data = build_train_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        # train
        T_model = x.Model(config).to(config.device)
        teacher_train(config, T_model, train_iter, dev_iter, test_iter)

    data = pd.read_csv('label.csv', encoding="utf-8", header=None, usecols=[0, 1, 2])
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.dropna()
    data.columns = ['user', 'text', 'pred']
    data = data.apply(pd.to_numeric, errors='ignore')

    student_train(data)

