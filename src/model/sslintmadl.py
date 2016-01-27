#!/usr/bin/env python
# coding=utf-8
"""
> File Name: sslintmadl.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 三  1/27 21:25:12 2016
"""

import sys
import logging
import numpy as np

import evaluate
import methods.scikitTSVM 
sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.splitvalue as split
import feature.decomposition as decomposition
sys.path.insert (0, '../..')
from configure import *

def s3vm_solver(train_data, train_label, validation, test, unlabel, feature_extract, feature_handler):
    """
    """
    logging.info('begin to train the s3vm classifier')

    #unlabel = unlabel[:100,:]
    #train_data = train_data[:100,:]
    #validation = validation[:100,:]
    #test = test[:100,:]
    #train_label = train_label[:100]

    train_data, validation, test , unlabel = feature_extract (train_data, train_label, validation, test, unlabel)
    # print new_train_data.shape
    train_data, validation, test , unlabel = feature_handler (train_data, validation, test, unlabel)

    data = np.vstack ([train_data, unlabel])
    label = np.hstack ([train_label, [-1] * unlabel.shape[0]])
    assert data.shape[0] == len (label)
    s3vm = methods.scikitTSVM.SKTSVM(kernel='linear')

    s3vm.fit (data , label)
    evaluate.get_auc (s3vm.predict_proba (validation)[:,1])
    return s3vm.predict_proba (train_data)[:,1]


if __name__ == '__main__' :
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()
    assert data.shape[0] == train_number + test_number + val_number + unlabel_number
    predict = s3vm_solver (data[:train_number,:], label, data[train_number:train_number+val_number,:], data[train_number+val_number:-unlabel_number,:], data[-unlabel_number:,:], decomposition.gbdt_dimreduce_threshold, split.undo)

    evaluate.output (uid, predict, ROOT + '/result/s3vm.csv')
