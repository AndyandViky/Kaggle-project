# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: datasets.py
@time: 2019/8/19 上午11:50
@desc: 
"""
from torch.utils.data import Dataset
from utils import csv2tensor
import pandas as pd
import torch


class DigitData(Dataset):

    def __init__(self, datapath, transform=None, train=True):
        # read csv file data
        self.datapath = datapath
        self.transform = transform
        self.datas = csv2tensor(pd.read_csv(self.datapath, skiprows=1))
        self.train = train

        train_length = int(len(self.datas) * 0.9)
        if self.train:
            self.image, self.label = self.datas[:train_length, 1:], self.datas[:train_length, 0]
        else:
            self.image, self.label = self.datas[train_length:, 1:], self.datas[train_length:, 0]

        # free some memory space
        del self.datas

    def __getitem__(self, index):

        img, target = self.image.data[index], int(self.label.data[index])
        img = img.view(28, 28)
        # img = Image.fromarray(img.numpy(), mode='L')

        # plt.imshow(img)
        # plt.show()

        if self.transform is not None:
            img = self.transform(img.numpy())

        return torch.as_tensor(img, dtype=torch.float32), target

    def __len__(self):

        return len(self.image)
