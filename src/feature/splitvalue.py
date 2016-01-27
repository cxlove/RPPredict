#!/usr/bin/env python
# coding=utf-8
"""
> File Name: splitvalue.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 二  1/19 12:55:21 2016
"""
import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, '..')
import utils.io as io 
import feature.convert as convert

sys.path.insert(0, '../..')
from configure import *

def split_value (val) :
    """
    convert val into the corresponding interval's index
    @params:
        val: the numerical value need to handle
    @return:
        $1: the index from 0 to SPLITCONTINUUM - 1
    """
    global min_val, max_val
    block_length = (max_val - min_val) / SPLITCONTINUUM
    if block_length < 0.01 : return 0
    block_index = int ((val - min_val) / block_length)
    if block_index < 0 : block_index = 0
    if block_index >= SPLITCONTINUUM : block_index = SPLITCONTINUUM - 1
    return block_index

def split_continuum_value_data (data) :
    """
    split the continuum value into some interval with same length
    then convert the category variable into binary variable 
    @params:
        data: original data (ndarray)
    @return:
        $1: the corresponding data after spliting
    """
    logging.info ('begin split_continuum_value_data')
    print data.shape
    if os.path.exists (ROOT + '/data/split_' + str (SPLITCONTINUUM)) :
        logging.info (ROOT + '/data/split_' + str (SPLITCONTINUUM) + ' exist!')
        return io.grab (ROOT + '/data/split_' + str (SPLITCONTINUUM))
    else :
        data = pd.DataFrame (data)
        feature_list = data.columns
        for feature in feature_list :
            global min_val, max_val
            min_val = min (data[feature].values)
            max_val = max (data[feature].values)
            data[feature] = data[feature].map (lambda x : split_value (x))
            data = convert.binary_feature (data, feature)
            data.drop (feature, axis = 1, inplace = 1)

        io.store (data.values[:,1:], ROOT + '/data/split_' + str (SPLITCONTINUUM))

    return data.values[:,1:]

def split_continuum_value_tvt (train, validation, test, unlabel) :
    """
    split the continuum value into some interval with same length
    then convert the category variable into binary variable 
    @params:
        train, validation, test: original data (ndarray)
    @return:
        $1, $2, $3: the corresponding train, validation, test set
    """
    data = np.vstack ([train, validation, test, unlabel])
    data = split_continuum_value_data (data)

    new_train = data[:train.shape[0],:]
    new_val = data[train.shape[0]:train.shape[0]+validation.shape[0],:]
    new_test = data[train.shape[0]+validation.shape[0]:-unlabel.shape[0],:]
    new_unlabel = data[-unlabel.shape[0]:,:]
    return new_train, new_val, new_test, new_unlabel

def undo (train, validation, test, unlabel) :
    """
    nothing to do
    """
    return train, validation, test, unlabel

