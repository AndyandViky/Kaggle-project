# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: visualization.py
@time: 2019/8/21 下午2:38
@desc: visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import TRAIN_PATH, TEST_PATH

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 根据 年龄 绘制 幸存 柱状图
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# 根据 年龄，等级 绘制 幸存 柱状图
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', bins=20, alpha=.5)
# grid.add_legend()

# 根据 等级，登船港口，性别 绘制 幸存 折线图
# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()

# 根据 性别，登船港口，是否幸存 绘制 票价 柱状图
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

plt.show()
