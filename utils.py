# -*- coding: utf-8 -*-
"""
# @Time    : 2020/5/19 下午3:52
# @Author  : HOY
# @Email   : huangouyan@changingedu.com
# @File    : utils.py
# @Software: PyCharm
"""
import time
from datetime import timedelta


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
