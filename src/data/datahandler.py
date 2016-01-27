#!/usr/bin/env python
# coding=utf-8
"""
> File Name: datahandler.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 四  1/14 14:42:30 2016
"""

import os
import sys
import logging
import pandas as pd

import read
sys.path.insert(0, '..')
import feature.decomposition as decomposition
import feature.convert as convert
import utils.io as io

sys.path.insert(0, '../..')
from configure import *

logging.basicConfig (level = logging.INFO,
                     format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                     datefmt = '%a, %d %b %Y %H:%M:%S',
                     filemode = 'w')


def feature_handler (data) :
    """
    convert categorical variable into binary variable and standardization dataset
    @parameters:
        data: original data
    @return:
        $1: the data after handlering 
    """
    logging.info ('begin to handle feature')
    featuretype = open (ROOT + '/data/features_type.csv').readlines ()
    for i in xrange (1 , len (featuretype)) :
        line = featuretype[i].strip ().split (',') 
        # remove the " in text
        line[0] = line[0][1:-1]
        line[1] = line[1][1:-1]
        # if the feature is categorical variable, convert it into binary variable
        if line[1] == 'category':
            data = convert.binary_feature (data , line[0])
            data.drop (line[0] , axis = 1 , inplace = 1)
        
    # standardization all of the feature
    featurelist = data.columns 
    for feature in featurelist :
        data = convert.scale_feature (data, feature)
        data.drop (feature, axis = 1, inplace = 1)
    logging.info ('finished hanlering feature')
    return data


def clean_data (usePCA = False) :
    """
    """
    logging.info ('begin to clean the data')
    if os.path.exists (ROOT + '/data/cleandata.csv') :
        # we need not to clean the data each time
        # if you want to reclean the data, please delete '../data/cleandata.csv' file
        logging.info ('the clean data is already exists')
        data = pd.read_csv (ROOT + '/data/cleandata.csv')
        train_number, val_number, test_number, unlabel_number, label, uid = io.grab (ROOT + '/data/datadescribe')
    else :
        data, train_number, val_number, test_number, unlabel_number, label, uid = read.read_data ()
        data = feature_handler (data)
        # store the result
        data.to_csv (ROOT + '/data/cleandata.csv')
        io.store ([train_number, val_number, test_number, unlabel_number, label, uid], ROOT + '/data/datadescribe')

    logging.info ('finished cleaning the data')

    if usePCA :
        # dimensionality reduction
        if not os.path.exists (ROOT + '/data/datapca') :
            # we need not to rerun this step
            # if you change the parameters and want to relearn it, please delete '../data/datapca' file
            data_values = decomposition.pca_solver (data)
            io.store (data_values, ROOT + '/data/datapca')

        data_values = io.grab (ROOT + '/data/datapca')
    else :
        data_values = data.values[:,1:]
    return data_values, train_number, val_number, test_number, unlabel_number, label, uid
    
if __name__ == '__main__' :
    clean_data ()
