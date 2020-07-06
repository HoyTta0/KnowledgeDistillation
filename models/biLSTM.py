# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/18 下午5:27
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : biLSTM.py
# @Software: PyCharm
"""
import torch.nn as nn
import torch.nn.functional as F


class biLSTM(nn.Module):
    def __init__(self):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=256,
                            num_layers=1, batch_first=True, dropout=0, bidirectional=True)
        # self.linear = nn.Linear(in_features=256, out_features=2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)     # LSTM 的返回很多
        out = self.fc1(lstm_out)
        activated_t = F.relu(out)
        linear_out = self.fc2(activated_t)

        return linear_out, hidden
