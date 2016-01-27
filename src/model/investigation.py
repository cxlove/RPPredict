#!/usr/bin/env python
# coding=utf-8
"""
> File Name: investigation.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 五  1/22 12:13:00 2016
"""
import sys
from sklearn.metrics import roc_auc_score

import logistic
sys.path.insert(0, '..')
import data.datahandler as datahandler
import feature.decomposition as decomposition
import feature.splitvalue as split

sys.path.insert(0, '../..')
from configure import *

def find_threshold_split_pos_neg () :
    """
    """
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()
    predict = logistic.lr_solver (data[:train_number,:], label, data[train_number:-test_number,:], data[-test_number:,:], decomposition.gbdt_dimreduce_threshold, split.split_continuum_value_tvt)
    print roc_auc_score (label, predict)
    neg = []
    pos = []
    for i in xrange (len (label)) :
        if label[i] == 0 :
            neg.append (predict[i])
        else :
            pos.append (predict[i])
    predict.sort ()
    pos.sort ()
    neg.sort ()
    with open (ROOT + '/result/pos', 'w') as out :
        for each in pos :
            out.write (str (each) + '\n')
    with open (ROOT + '/result/neg', "w") as out :
        for each in neg :
            out.write (str (each) + '\n')

    print predict[len (neg)]

    # 0.803715175861


if __name__ == '__main__' :
    find_threshold_split_pos_neg ()
    
     
