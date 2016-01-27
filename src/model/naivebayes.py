#!/usr/bin/env python
# coding=utf-8
"""
> File Name: naivebayes.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 三  1/20 14:19:47 2016
"""
import sys
import logging
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

import evaluate
sys.path.insert(0, '..')
import feature.splitvalue as split
import feature.decomposition as decomposition
import data.datahandler as datahandler

sys.path.insert(0, '../..')
from configure import *

def nb_solver(train_data, train_label, validation, test, classifier, dimreduce, convertbinary):
    """
    """
    logging.info('begin to train the naive bayes classifier')

    # train_data = train_data[:100,:]
    # validation = validation[:100,:]
    # test = test[:100,:]
    # train_label = train_label[:100]
    train_data, validation, test = dimreduce(train_data, train_label, validation, test)
    # print new_train_data.shape
    train_data, validation, test = convertbinary(train_data, validation, test)

    nb = classifier ()
    nb.fit(train_data , train_label)
    evaluate.get_auc (nb.predict_proba (validation)[:,1])
    return nb.predict_proba (test)[:,1]

if __name__ == "__main__" :
    data, train_number, val_number, test_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number
    predict = nb_solver (data[:train_number,:], label, data[train_number:-test_number,:], data[-test_number:,:], BernoulliNB, decomposition.undo, split.undo)

    evaluate.output (uid, predict, ROOT + '/result/naivebayes.csv')




