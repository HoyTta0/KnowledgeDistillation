# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 上午9:55
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : fullyConnect.py
# @Software: PyCharm
"""
import torch.nn as nn
import torch.nn.functional as F


class fullCon(nn.Module):
    def __init__(self):
        super(fullCon, self).__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output