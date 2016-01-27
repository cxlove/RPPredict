#!/usr/bin/env python
# coding=utf-8
"""
> File Name: logistic.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 五  1/15 21:01:16 2016
"""
import sys
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV 
from sklearn.externals import joblib

import evaluate
sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.splitvalue as split
import feature.decomposition as decomposition

sys.path.insert (0, '../..')
from configure import *

def lr_solver(train_data, train_label, validation, test, unlabel, feature_extract, feature_handler):
    """
    """
    logging.info('begin to train the lr classifier')

    # train_data = train_data[:100,:]
    # validation = validation[:100,:]
    # test = test[:100,:]
    # train_label = train_label[:100]
    train_data, validation, test , unlabel = feature_extract (train_data, train_label, validation, test, unlabel)
    # print new_train_data.shape
    train_data, validation, test , unlabel = feature_handler (train_data, validation, test, unlabel)
    """
    lr = LogisticRegression ()
    params_test = {"penalty":['l1','l2'],
                 "C":[0.1,0.2,0.3,0.5,0.7,1,3,5],
                 "tol":[0.001,0.003,0.005,0.01,0.05,0.1,0.5],
                 "random_state":[1000000007]}
    rand_search_result = GridSearchCV (lr, param_grid = params_test, n_jobs = 3, cv = 3, scoring='roc_auc')
    rand_search_result.fit (train_data , train_label)
    params = evaluate.report (rand_search_result.grid_scores_)
    print params
    """

    print train_data.shape[1]
    params = {'penalty': 'l1', 'C':0.1 , 'random_state': 1000000007, 'tol': 0.001, 'warm_start' : True}

    lr = LogisticRegression(**params)    
    lr.fit (train_data , train_label)
    joblib.dump (lr, ROOT + '/result/lr.pkl')
    evaluate.get_auc (lr.predict_proba (validation)[:,1])
    return lr.predict_proba (train_data)[:,1]

if __name__ == "__main__" :
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number + unlabel_number
    predict = lr_solver (data[:train_number,:], label, data[train_number:train_number+val_number,:], data[train_number+val_number:-unlabel_number,:], data[-unlabel_number:,:],  decomposition.gbdt_dimreduce_threshold, split.undo) 

    evaluate.output (uid, predict, ROOT + '/result/lr.csv')
    


