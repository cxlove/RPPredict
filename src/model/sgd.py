#!/usr/bin/env python
# coding=utf-8
"""
> File Name: sgd.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 一  1/25 19:12:06 2016
"""
import sys
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

import evaluate
sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.splitvalue as split
import feature.decomposition as decomposition

sys.path.insert (0, '../..')
from configure import *

def sgd_solver(train_data, train_label, validation, test, unlabel, feature_extract, feature_handler):
    """
    """
    logging.info('begin to train the sgd classifier')

    # train_data = train_data[:100,:]
    # validation = validation[:100,:]
    # test = test[:100,:]
    # train_label = train_label[:100]
    train_data, validation, test , unlabel = feature_extract (train_data, train_label, validation, test, unlabel)
    # print new_train_data.shape
    train_data, validation, test , unlabel = feature_handler (train_data, validation, test, unlabel)

    sgd = SGDClassifier(loss = 'modified_huber', alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
                     eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                     learning_rate='optimal', n_iter=5, n_jobs=2,
                     penalty='l2', power_t=0.5, random_state=1000000007, shuffle=True,
                     verbose=0, warm_start=True)
    sgd.fit (train_data , train_label)
    joblib.dump (sgd, ROOT + '/result/sgd.pkl')
    evaluate.get_auc (sgd.predict_proba (validation)[:,1])
    return sgd.predict_proba (train_data)[:,1]

if __name__ == "__main__" :
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number + unlabel_number
    predict = sgd_solver (data[:train_number,:], label, data[train_number:train_number+val_number,:], data[train_number+val_number:-unlabel_number,:], data[-unlabel_number:,:],  decomposition.gbdt_dimreduce_threshold, split.undo) 

    evaluate.output (uid, predict, ROOT + '/result/sgd.csv')
    


