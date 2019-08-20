# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: config.py.py
@time: 2019/8/19 上午11:48
@desc: config
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(ROOT_DIR, 'model')

LOG_DIR = os.path.join(ROOT_DIR, 'log')

DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

parameter_grid = {
    'max_depth' : [4, 6, 8],
    'n_estimators': [50, 10],
    'max_features': ['sqrt', 'auto', 'log2'],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [True, False],
}

# AdaBoost parameters
ada_params = {
    'n_estimators': [400, 500, 600],
    'learning_rate': [0.5, 0.75, 0.9]
}

xgb_params = {
    "n_estimators": [2000, 2100],
    "max_depth": [4, 5, 6],
    "min_child_weight": [2, 3, 10],
    #gamma=1,
    "gamma": [0.9, 1.0, 1.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "objective": ['binary:logistic'],
    "nthread": [-1],
    "scale_pos_weight": [1]
}

