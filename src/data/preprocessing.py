#!/usr/bin/env python
# coding=utf-8
"""
> File Name: preprocessing.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 五  1/22 19:36:55 2016
"""
import sys

import datahandler
sys.path.insert(0, '..')
import feature.decomposition as decomposition
sys.path.insert(0, '../..')
import feature.splitvalue as split
import model.evaluate as evaluate


if __name__ == '__main__' :
    data, train_number, val_number, test_number, unlabel_number, label, uid = datahandler.clean_data ()
    train = data[:train_number,:]
    validation = data[train_number:train_number+val_number,:]
    test = data[train_number+val_number:-unlabel_number,:]
    unlabel = data[-unlabel_number:,:]
    print train.shape, validation.shape, test.shape, unlabel.shape

    train, validation, test, unlabel = decomposition.gbdt_dimreduce_threshold (train, label, validation, test, unlabel)
    train, validation, test, unlabel = split.split_continuum_value_tvt (train, validation, test, unlabel)


    """
    train: traning set
    validation: validation set, used for testing, the label is in 'val_cv_y.csv'
    test: testing set
    unlabel: unlabel set
    label: the label of train
    uid: the uid for testing
    
    if you want to calulate the auc score in validation set
    evaluate.get_auc (predict)   # predict is the list or array for your prediction in validation set 
    such as : evaluate.get_auc (lr.predict_proba (validation)[:,1])

    if you want to save the final result for predicting
    evaluate.output (uid, predict, path)  # uid is the id for testing, predict is your prediction in testing set, path is the path you want to save the result file

    """

    
    
