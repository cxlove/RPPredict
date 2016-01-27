#!/usr/bin/env python
# coding=utf-8
"""
> File Name: svm.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 五  1/15 19:36:38 2016
"""
import sys
import logging
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV , GridSearchCV

sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.decomposition as decomposition
import feature.splitvalue as split

sys.path.insert(0, '../..')
from configure import *

def svm_solver(train_data, train_label, validation, test, dimreduce, convertbinary) :
    """
    """
    logging.info ('begin to train the svm classifier')

    # train_data = train_data[:100,:]
    # validation = validation[:100,:]
    # test = test[:100,:]
    # train_label = train_label[:100]
    train_data, validation, test = dimreduce(train_data, train_label, validation, test)
    # print new_train_data.shape
    train_data, validation, test = convertbinary(train_data, validation, test)

    """
    svc = SVC ()
    params_rbf = {"kernel": ['rbf'],
             "class_weight": ['auto'],
             "C": [0.1 ,0.2 ,0.3 ,0.5 ,1, 2, 3, 5, 10],
             "gamma": [0.01, 0.03,  0.05, 0.1, 0.2, 0.3, 0.5],
             "tol": 10.0** -np.arange(1, 5),
             "random_state": [1000000007]}
    logging.info ("Hyperparameter opimization using RandomizedSearchCV...")
    rand_search_result = RandomizedSearchCV (svc, param_distributions = params_rbf, n_jobs = -1, cv = 3, n_iter = 30)
    # rand_search_result = GridSearchCV (svc , param_grid = params_rbf , n_jobs = 8  , cv = 3)
    rand_search_result.fit (train_data , train_label)
    params = tools.report (rand_search_result.grid_scores_)
    """
    params = {'kernel': 'poly', 'C': 0.1, 'random_state': 1000000007, 'tol': 0.001, 'gamma': 0.1, 'class_weight': 'auto'}
    svc = SVC (probability = True, **params)

    svc.fit (train_data , train_label)
    evaluate.get_auc (svc.predict_proba (validation)[:,1])
    return svc.predict_proba (test)[:,1]

if __name__ == "__main__" :
    data, train_number, val_number, test_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number
    predict = svm_solver (data[:train_number,:], label, data[train_number:-test_number,:], data[-test_number:,:],  decomposition.gbdt_dimreduce_threshold, split.split_continuum_value_tvt)

    evaluate.output (uid, predict, ROOT + '/result/svm.csv')
