# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: config.py
@time: 2019/8/19 上午10:31
@desc: 
"""

import os
import torch

Digit_DIR = os.path.dirname(os.path.abspath(__file__))

Data_DIR = os.path.join(Digit_DIR, 'datasets')

Model_DIR = os.path.join(Digit_DIR, 'model')

Log_DIR = os.path.join(Digit_DIR, 'log')

train_data_path = os.path.join(Data_DIR, 'train.csv')

test_data_path = os.path.join(Data_DIR, 'test.csv')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")