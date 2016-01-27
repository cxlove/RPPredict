#!/usr/bin/env python
# coding=utf-8
"""
> File Name: sslinsklearn.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 五  1/22 15:52:17 2016
"""

import sys
import numpy as np
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

import evaluate
sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.decomposition as decomposition

sys.path.insert(0, '../..')
from configure import *

def ssl_solver (train, label, validation, test, unlabel, dimreduce, classifier = LabelSpreading) :
    """
    """

    train, validation, test, unlabel = dimreduce (train, label, validation, test, unlabel)

    data = np.vstack ([train, unlabel])
    label = np.hstack ([label, [-1] * unlabel.shape[0]])
    assert data.shape[0] == len (label)

    cf = classifier (kernel = 'knn', n_neighbors = 100, max_iter = 3)
    # cf = classifier (kernel = 'rbf', gamma = 0.3, max_iter = 3)

    cf.fit (data, label)
    evaluate.get_auc (cf.predict_proba (validation)[:,1])
    return cf.predict_proba (test)[:,1]

if __name__ == '__main__' :
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number + unlabel_number
    predict = ssl_solver (data[:train_number,:], label, data[train_number:train_number + val_number,:], data[train_number+val_number:-unlabel_number,:], data[-unlabel_number:,:], decomposition.gbdt_dimreduce_threshold)
    evaluate.output (uid, predict, ROOT + '/result/ssl.csv')
