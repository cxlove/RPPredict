#!/usr/bin/env python
# coding=utf-8
"""
> File Name: knn.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 三  1/20 10:50:02 2016
"""

import sys
import logging
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import evaluate
sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.splitvalue as split
import feature.decomposition as decomposition

sys.path.insert (0, '../..')
from configure import *


def knn_solver(train_data, train_label, validation, test, dimreduce, convertbinary):
    """
    """
    logging.info('begin to train the knn classifier')

    # train_data = train_data[:100,:]
    # validation = validation[:100,:]
    # test = test[:100,:]
    # train_label = train_label[:100]
    train_data, validation, test = dimreduce(train_data, train_label, validation, test)
    # print new_train_data.shape
    # train_data, validation, test = convertbinary(train_data, validation, test)

    knn = KNeighborsClassifier (algorithm = 'auto', n_neighbors = 10, p = 3)
    knn.fit (train_data , train_label)
    tools.get_auc (knn.predict_proba (validation)[:,1])
    return knn.predict_proba (test)[:,1]

if __name__ == "__main__" :
    data, train_number, val_number, test_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number
    predict = knn_solver (data[:train_number,:], label, data[train_number:-test_number,:], data[-test_number:,:],  decomposition.gbdt_dimreduce_threshold, split.split_continuum_value_tvt)

    evaluate.output (uid, predict, ROOT + '/result/knn.csv')


