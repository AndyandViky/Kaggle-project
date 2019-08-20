# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: xgboost-train.py
@time: 2019/8/19 下午7:10
@desc: use xgboost to train
"""

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from datasets import filterdata
from config import TEST_PATH, TRAIN_PATH, xgb_params


class Trainer:

    def __init__(self):
        super(Trainer, self).__init__()

    def train(self):

        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        data, target, id = filterdata(train, test)

        train_data = data[:891]
        test_data = data[891:]

        gbm = xgb.XGBClassifier()

        cross_validation = StratifiedKFold(n_splits=5)

        grid_search = GridSearchCV(gbm,
                                   scoring="accuracy",
                                   param_grid=xgb_params,
                                   cv=cross_validation,
                                   verbose=1
                                   )

        grid_search.fit(train_data, target)
        best_parameters = grid_search.best_params_

        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(best_parameters))

        model = xgb.XGBClassifier(**best_parameters)
        model.fit(train_data, target)

        output = model.predict(test_data).astype(int)

        submission = {
            'PassengerId': id,
            'Survived': output
        }
        solution = pd.DataFrame(submission)
        solution.head()

        solution.to_csv('submission.csv', index=False)


if __name__ == "__main__":

    trainer = Trainer()
    trainer.train()