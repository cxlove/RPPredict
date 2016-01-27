#!/usr/bin/env python
# coding=utf-8
"""
> File Name: gbdt.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 五  1/15 20:54:00 2016
"""

import sys
import logging
from sklearn.grid_search import RandomizedSearchCV , GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

import evaluate
sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.decomposition as decomposition

sys.path.insert(0, '../..')
from configure import *

def gbdt_solver (train_data, train_label, validation, test , unlabel, dimreduce = decomposition.undo) :
    """
    """
    # train_data = train_data[:100,:]
    # train_label = train_label[:100]

    logging.info ('begin to train the gbdt classifier')
    new_train_data, new_val, new_test, new_unlabel = dimreduce(train_data, train_label, validation, test, unlabel)
    logging.info ('finished feature extracting')

    """
    gb = GradientBoostingClassifier ()
    params_gbdt = {"n_estimators":[100,200,500,1000],
                 "learning_rate":[0.02,0.03,0.05,0.1],
                 "max_depth":[3,5,7,9],
                 "random_state":[1000000007]}"""

    # rand_search_result = GridSearchCV (gb, param_grid = params_gbdt , n_jobs = 3  , cv = 3, scoring = 'roc_auc')
    # rand_search_result = RandomizedSearchCV (gb, param_distributions = params_gbdt, n_jobs = 3, cv = 3, n_iter = 100, scoring = 'roc_auc')
    # rand_search_result.fit (new_train_data , train_label)
    # params = tools.report (rand_search_result.grid_scores_)

    params = {'n_estimators': 600, 'learning_rate': 0.03, 'random_state': 1000000007, 'max_depth': 2 , 'warm_start' : True}
    gb = GradientBoostingClassifier (**params)
    gb.fit (new_train_data , train_label)
    joblib.dump (gb, ROOT + '/result/gbdt.pkl')
    evaluate.get_auc (gb.predict_proba (new_val)[:,1])
    return gb.predict_proba (new_test)[:,1]

    

if __name__ == "__main__" :
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number + unlabel_number
    predict = gbdt_solver (data[:train_number,:], label, data[train_number:train_number+val_number,:], data[train_number+val_number:-unlabel_number,:], data[-unlabel_number:,:], decomposition.gbdt_dimreduce_threshold)

    evaluate.output (uid, predict, ROOT + '/result/gbdt.csv')
    

    


