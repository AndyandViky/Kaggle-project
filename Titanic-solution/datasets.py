# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: datasets.py.py
@time: 2019/8/19 上午11:48
@desc: datasets
"""

import pandas as pd


def filterdata(train, test):
    """
    filter origin data:
        1. replacing missing values
        2. change some data encoding in dummy variable
        3. add some extend data
    :param train:
    :param test:
    :return: data, target, PassengerId
    """

    PassengerId = test.PassengerId.values

    target = train.Survived
    train.drop(['Survived'], 1, inplace=True)

    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'PassengerId'], 1, inplace=True)

    # fix missing values
    combined.Age.fillna(train.Age.median(), inplace=True)
    combined.Fare.fillna(train.Fare.mean(), inplace=True)
    combined.Embarked.fillna(train.Embarked.mode()[0], inplace=True)

    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix="Embarked")
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop("Embarked", 1, inplace=True)

    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    combined['Title'] = combined['Title'].map({
        "Capt": "Other",
        "Col": "Other",
        "Major": "Other",
        "Jonkheer": "Other",
        "Don": "Other",
        "Dona": "Other",
        "Sir": "Other",
        "Lady": "Other",
        "Dr": "Other",
        "Rev": "Other",
        "the Countess": "Other",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master"
    })
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)
    # removing the name and title variable
    combined.drop(columns=['Name', 'Title'], axis=1, inplace=True)

    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)
    # removing "Cabin"
    combined.drop('Cabin', axis=1, inplace=True)

    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})

    # encoding "Pclass" into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    # adding dummy variable
    combined = pd.concat([combined, pclass_dummies], axis=1)
    # removing "Pclass"
    combined.drop('Pclass', axis=1, inplace=True)

    def cleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    # Extracting dummy variables from tickets:
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    return combined, target, PassengerId




