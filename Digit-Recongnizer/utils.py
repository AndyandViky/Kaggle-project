# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: utils.py
@time: 2019/8/19 上午11:47
@desc: 
"""

import torch
import torch.nn as nn


# util functions
def csv2tensor(data):
    return torch.from_numpy(data.to_numpy())


def init_weights(Net):
    for net in Net.modules():
        if isinstance(net, nn.Conv2d):
            net.weight.data.normal_(0, 0.02)
            net.bias.data.zero_()
        if isinstance(net, nn.Linear):
            net.weight.data.normal_(0, 0.02)
            net.bias.data.zero_()


def caculate_accuracy(pred, target):

    correct = 0
    correct += (pred == target).sum()

    acc = correct.numpy() / len(pred.numpy())
    return acc