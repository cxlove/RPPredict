#!/usr/bin/env python
# coding=utf-8
"""
> File Name: convert.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 四  1/21 19:34:59 2016
"""

import logging
import pandas as pd
from sklearn import preprocessing

def binary_feature (data, feature) :
    """
    convert categorical variable into dummy/binary variable by pandas.get_dummies
    @parameters:
        data: original data
        feature: the feature need to solve
    @return:
        $1: the data after handlering
    """
    logging.info ('convert to binary variable for feature: %s' % feature)
    data = pd.concat ([data , pd.get_dummies (data[feature]).rename (columns = lambda x : str (feature) + "_" + str (x))] , axis = 1)
    return data

def scale_feature (data, feature) :
    """
    standardization of datasets
    make the data look like standard normally distributed data: Gaussian with zero mean and unit variance.
    @parameters:
        data: original data
        feature: the feature need to solve
    @return:
        $1: the data after handlering
    """
    logging.info ('standardization for feature: %s' % feature)
    scaler = preprocessing.StandardScaler ()
    data[feature + '_scaled'] = scaler.fit_transform (data[feature])
    return data

