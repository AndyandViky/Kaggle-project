# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: model.py.py
@time: 2019/8/19 上午11:48
@desc: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import init_weights


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )


class SpacialTransformer(nn.Module):

    def __init__(self):
        super(SpacialTransformer, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),

            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):

        return self.stn(x)


class Classifier_CNN(nn.Module):

    def __init__(self, feature_dim=784, latent_dim=10, input_size=(1, 28, 28), verbose=False):
        super(Classifier_CNN, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.cshape = (128, 6, 6)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.vervose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        self.model = nn.Sequential(
            self.model,
            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, self.latent_dim),
            # nn.Softmax(),
        )

        init_weights(self)

        if self.vervose:
            print(self.model)

    def forward(self, x):

        return self.model(x)


class ResNet_Block(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ResNet_Block, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),

            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),
        )

    def forward(self, x):

        return x + self.model(x)


class RES_NET_Classifier(nn.Module):
    def __init__(self, feature_dim=784, latent_dim=10, input_size=(1, 28, 28), verbose=False):
        super(RES_NET_Classifier, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.cshape = (128, 1, 1)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.vervose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        res_block = []
        for i in range(3):
            res_block += [
                ResNet_Block(64, 64)
            ]

        res_block += [
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        ]

        for i in range(3):
            res_block += [
                ResNet_Block(128, 128)
            ]

        self.model = nn.Sequential(
            self.model,
            *res_block,
            nn.AvgPool2d(2, 2),

            Reshape(self.lshape),

            # nn.Linear(self.iels, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),

            nn.Linear(self.iels, self.latent_dim),
        )

        init_weights(self)

        if self.vervose:
            print(self.model)

    def forward(self, x):

        return self.model(x)