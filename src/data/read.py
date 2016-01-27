#!/usr/bin/env python
# coding=utf-8
"""
> File Name: read.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 四  1/21 19:39:09 2016
"""
import sys
import logging
import pandas as pd

sys.path.insert(0, '../..')
from configure import *

def read_data () :
    """
    read the original data from csv file 
    training data: ROOT + /data/train_cv_x.csv
    training label: ROOT + /data/train_cv_y.csv
    validation data: ROOT + /data/val_cv_x.csv
    testing data: ROOT + /data/test_cv_x.csv 
    @return: 
        $1: the whole data after cleaning (ndarray)
        $2: the number of traning instances (integer)
        $3: the number of validation instances (integer)
        $4: the number of testing instances (integer)
        $5: the label of traning set (ndarray)
        $6: the user id of testing instances (ndarray)
    """
    logging.info ('begin to read data')
    train = pd.read_csv (ROOT + "/data/train_cv_x.csv")
    val = pd.read_csv (ROOT + "/data/val_cv_x.csv")
    label = pd.read_csv (ROOT + "/data/train_cv_y.csv")
    test = pd.read_csv (ROOT + "/data/test_x.csv")
    unlabel = pd.read_csv (ROOT + "/data/train_unlabeled.csv")
    print train.shape
    print val.shape
    print test.shape
    print unlabel.shape
    # train = pd.merge (train , label , on = 'uid')
    data = pd.concat ([train , val , test, unlabel], ignore_index = True)
    data.reset_index (inplace = True)
    data.drop ('index', axis = 1, inplace = 1)
    data.drop ('uid', axis = 1, inplace = 1)
    # data.drop ('y', axis = 1 , inplace = 1)
    logging.info ('finished reading data')
    return data, train.shape[0], val.shape[0] , test.shape[0], unlabel.shape[0], label.y.values, test.uid.values

if __name__ == '__main__' :
    read_data ()
