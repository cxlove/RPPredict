#!/usr/bin/env python
# coding=utf-8
"""
> File Name: randomforest.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 一  1/25 19:12:06 2016
"""
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import evaluate
sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.splitvalue as split
import feature.decomposition as decomposition

sys.path.insert (0, '../..')
from configure import *

def rf_solver(train_data, train_label, validation, test, unlabel, feature_extract, feature_handler):
    """
    """
    logging.info('begin to train the random forest classifier')

    # train_data = train_data[:100,:]
    # validation = validation[:100,:]
    # test = test[:100,:]
    # train_label = train_label[:100]
    train_data, validation, test , unlabel = feature_extract (train_data, train_label, validation, test, unlabel)
    # print new_train_data.shape
    train_data, validation, test , unlabel = feature_handler (train_data, validation, test, unlabel)

    rf = RandomForestClassifier (warm_start = True, n_jobs = 2, n_estimators = 2000, max_depth = 3, min_samples_split = 50)
    rf.fit (train_data , train_label)
    # joblib.dump (rf, ROOT + '/result/rf.pkl')
    evaluate.get_auc (rf.predict_proba (validation)[:,1])
    return rf.predict_proba (train_data)[:,1]

if __name__ == "__main__" :
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number + unlabel_number
    predict = rf_solver (data[:train_number,:], label, data[train_number:train_number+val_number,:], data[train_number+val_number:-unlabel_number,:], data[-unlabel_number:,:],  decomposition.gbdt_dimreduce_threshold, split.undo) 

    evaluate.output (uid, predict, ROOT + '/result/rf.csv')
    


