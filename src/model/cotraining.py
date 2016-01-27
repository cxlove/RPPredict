#!/usr/bin/env python
# coding=utf-8
"""
> File Name: cotraining.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 五  1/22 13:00:56 2016
"""

import sys
import copy
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

import evaluate
sys.path.insert(0, '..')
import data.datahandler as datahandler
import utils.io as io
import feature.splitvalue as split
import feature.decomposition as decomposition

sys.path.insert(0, '../..')
from configure import *

def lr_solver () :
    params = {'penalty': 'l1', 'C':0.1 , 'random_state': 1000000007, 'tol': 0.001}
    lr = LogisticRegression (**params)
    return lr

def gbdt_solver () :
    params = {'n_estimators': 600, 'learning_rate': 0.03, 'random_state': 1000000007, 'max_depth': 2}
    gb = GradientBoostingClassifier (**params)
    return gb

def training (model_one, model_two, unlabel, unlabel_label, unlabel_index, train, label) :
    predict = model_one.predict_proba (unlabel[unlabel_index,:])[:,1]
    index = range (0, len (unlabel_index))
    index.sort (key = lambda v : predict[v])
    unlabel_index = [unlabel_index[i] for i in index]
    n = len (unlabel_index)
    X = []
    Y = []

    # select 500,   positive: negative = 9 : 1
    POS = 450
    NEG = 50

    r = max (0 , n - POS)
    l = min (r - 1, NEG - 1)
    for i in xrange (n - 1, r - 1, -1) :
        X.append (unlabel[unlabel_index[i]])
        unlabel_label[unlabel_index[i]] = 1
        Y.append (1)

    for i in xrange (0, l + 1) :
        X.append (unlabel[unlabel_label[i]])
        unlabel_label[unlabel_index[i]] = 0
        Y.append (0)

    assert (n - 1 - r + 1) > 0
    assert (l + 1) > 0
    assert len (X) == len (Y)

    # if type (model_two) is GradientBoostingClassifier :
    #    model_two.n_estimators += 10 

    train = np.vstack ([train, X])
    label = np.hstack ([label, Y])

    model_two.fit (train, label)
    if l + 1 <= r - 1 :
        unlabel_index = unlabel_index[l + 1: r]
    else : unlabel_index = []
    return model_one, model_two, unlabel_label, unlabel_index, train, label

def cotraining (model_one, model_two, n_iter = 100) :
    """
    """
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()

    train = data[:train_number,:]
    validation = data[train_number:train_number+val_number:,:]
    test = data[train_number+val_number:-unlabel_number,:]
    unlabel = data[-unlabel_number:,:]

    train, validation, test, unlabel = decomposition.gbdt_dimreduce_threshold (train, label, validation, test, unlabel) 
    # train, validation, test, unlabel = split.split_continuum_value_tvt (train, validation, test, unlabel)

#    train_number = 100
#    unlabel_number = 1000
#
#    train = train[:100,:]
#    unlabel = unlabel[:1000,:]
#    label = label[:100]

    train_one = copy.deepcopy (train)
    label_one = copy.deepcopy (label)
    train_two = copy.deepcopy (train)
    label_two = copy.deepcopy (label)

    model_one.fit (train_one, label_one)
    model_two.fit (train_two, label_two)

    for iter in xrange (1 , n_iter + 1 , 1) :
        logging.info ('#%d iter for co-training :' % iter)

        unlabel_label = [-1] * unlabel_number
        unlabel_index = range (0, unlabel_number)
        step = 0
        while len (unlabel_index) > 0 :
            step += 1
            logging.info ('co-training step #%d , reamining unlabel: %d' % (step, len (unlabel_index)))
            model_one, model_two, unlabel_label, unlabel_index, train_two, label_two = training (model_one, model_two, unlabel, unlabel_label, unlabel_index, train_two, label_two)
            model_two, model_one, unlabel_label, unlabel_index, train_one, label_one = training (model_two, model_one, unlabel, unlabel_label, unlabel_index, train_one, label_one)
            
            evaluate.get_auc (model_one.predict_proba (validation)[:,1])
            evaluate.get_auc (model_two.predict_proba (validation)[:,1])
            evaluate.get_auc ((model_one.predict_proba (validation)[:,1] + model_two.predict_proba (validation)[:,1]) / 2.0)

            joblib.dump (model_one, ROOT + '/result/model/model_one_%d_%d.pkl' % (iter, step))
            joblib.dump (model_two, ROOT + '/result/model/model_two_%d_%d.pkl' % (iter, step))
    
            evaluate.output (uid, (model_one.predict_proba (test)[:,1] + model_two.predict_proba (test)[:,1]) / 2.0, ROOT + '/result/predict/cotraining_%d_%d.csv' % (iter, step))
            evaluate.output (uid, model_one.predict_proba (test)[:,1], ROOT + '/result/predict/model_one_%d_%d.csv' % (iter, step))
            evaluate.output (uid, model_two.predict_proba (test)[:,1], ROOT + '/result/predict/model_two_%d_%d.csv' % (iter, step))
    

if __name__ == '__main__' :
#    lr = joblib.load (ROOT + '/result/lr.pkl')
#    logging.info ('load lr model done!')
#    gbdt = joblib.load (ROOT + '/result/gbdt.pkl')
#    logging.info ('load gbdt model done!')
#    cotraining (lr, gbdt)
    cotraining (lr_solver (), gbdt_solver ())
