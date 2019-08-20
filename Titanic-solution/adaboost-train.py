# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: adaboost-train.py
@time: 2019/8/20 上午9:45
@desc: adaboost
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from datasets import filterdata
from config import TEST_PATH, TRAIN_PATH, ada_params


class Trainer:

    def __init__(self):
        super(Trainer, self).__init__()

    def train(self):

        train_data = pd.read_csv(TRAIN_PATH)
        test_data = pd.read_csv(TEST_PATH)
        data, target, id = filterdata(train_data, test_data)

        train, test = data[:891], data[891:]

        ada = AdaBoostClassifier()
        cross_validation = StratifiedKFold(n_splits=5)

        grid_search = GridSearchCV(
            ada,
            cv=cross_validation,
            scoring="accuracy",
            param_grid=ada_params,
            verbose=1
        )

        grid_search.fit(train, target)
        best_parameters = grid_search.best_params_

        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(best_parameters))

        # model = AdaBoostClassifier(**best_parameters)
        # model.fit(train, target)
        #
        # output = model.predict(test)
        #
        # submission = {
        #     'PassengerId': id,
        #     'Survived': output
        # }
        # solution = pd.DataFrame(submission)
        # solution.head()
        #
        # solution.to_csv('submission.csv', index=False)


if __name__ == "__main__":

    trainer = Trainer()
    trainer.train()


