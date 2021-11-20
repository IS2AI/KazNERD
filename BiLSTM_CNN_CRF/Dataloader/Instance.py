# coding=utf-8
# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:56
# @File : Instance.py
# @Last Modify Time : 2018/1/30 15:56
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Instance.py
    FUNCTION : Data Instance
"""

import torch
import random

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Instance:
    """
        Instance
    """
    def __init__(self):
        self.words = []
        self.chars = []
        self.labels = []
        self.words_size = 0
        self.chars_size = 0

        self.words_index = []
        self.chars_index = []
        self.label_index = []


