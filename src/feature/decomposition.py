#!/usr/bin/env python
# coding=utf-8
"""
> File Name: decomposition.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 一  1/18 14:11:46 2016
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, '..')
import utils.io as io

sys.path.insert(0, '../..')
from configure import *

def pca_solver (data, K = PCACOMPONENT) :
    """
    Linear dimensionality reduction by pricipal component analysis 
    @parameters:
        data: original data (dataFrame)
    @return:
        $1: the data after handlering (ndarray)
    """
    logging.info ('begin to run pca')
    logging.info ('the number of components in pca is %d' % PCACOMPONENT)
    pca = PCA (n_components = K , whiten = True)
    if type (data) is pd.DataFrame :
        data_values = data.values[:,1:]
    else :
        data_values = data
    pca.fit (data_values)
    pca_data = pca.transform (data_values)
    logging.info ('finished pca')
    return pca_data 

def pca (train, label, validataion, test, unlabel) :
    """
    """
    pca_data = np.vstack ([train, validataion, test, unlabel])
    pca_data = pca_solver (pca_data)
    return pca_data[:train.shape[0],:], pca_data[train.shape[0]+validataion.shape[0]:-unlabel.shape[0],:], pca_data[-unlabel.shape[0]:,:]

def gbdt_feature_importance (train, label) :
    if os.path.exists (ROOT + '/data/feature_importance') :
        logging.info ('feature_importance exists!')
        feature_importance = io.grab (ROOT + '/data/feature_importance')
    else :
        logging.info ('feature_importance start!')
        gb = GradientBoostingClassifier (n_estimators = 500 , learning_rate = 0.05 , max_depth = 3 , random_state = 1000000007).fit (train, label)
        feature_importance = gb.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max ())
        io.store (feature_importance, ROOT + '/data/feature_importance')
    return feature_importance

def gbdt_dimreduce_threshold (train_data, train_label, validataion, test, unlabel, feature_threshold = GBDTFEATURETHRESHOLD) :
    """
    """
    logging.info ('begin gbdt_dimreduce_threshold')
    if os.path.exists (ROOT + '/data/gbdt_threshold_' + str (GBDTFEATURETHRESHOLD)) :
        logging.info (ROOT + '/data/gbdt_threshold_' + str (GBDTFEATURETHRESHOLD) + ' exist!')
        important_index, sorted_index = io.grab(ROOT + '/data/gbdt_threshold_' + str (GBDTFEATURETHRESHOLD))
    else :
        feature_importance = gbdt_feature_importance (train_data, train_label)
        important_index = np.where (feature_importance > feature_threshold)[0]
        sorted_index = np.argsort (feature_importance[important_index])[::-1]
        io.store ([important_index, sorted_index], ROOT + '/data/gbdt_threshold_' + str (GBDTFEATURETHRESHOLD))

    new_train_data = train_data[:,important_index][:,sorted_index]
    new_val = validataion[:,important_index][:,sorted_index]
    new_test = test[:,important_index][:,sorted_index]
    new_unlabel = unlabel[:,important_index][:,sorted_index]
    return new_train_data, new_val, new_test, new_unlabel
   
def gbdt_dimreduce_number (train_data, train_label, validataion, test, unlabel, K = GBDTFEATURENUMBER) :
    """  
    """
    logging.info ('before gbdt dim-reducing : (%d %d)' % (train_data.shape))
    if os.path.exists (ROOT + '/data/gbdt_number_' + str (K)) :
        logging.info (ROOT + '/data/gbdt_number_' + str (K) + ' exist!')
        sorted_index = io.grab (ROOT + '/data/gbdt_number_' + str (K)) 
    else :
        feature_importance = gbdt_feature_importance (train_data, train_label)
        sorted_index = np.argsort (feature_importance)[::-1]
        sorted_index = sorted_index[:K]
    # print 'feature importance :' , feature_importance[sorted_index]

    new_train_data = train_data[:,sorted_index]
    new_val = validataion[:,sorted_index]
    new_test = test[:,sorted_index]
    new_unlabel = label[:,sorted_index]
    logging.info ('after gbdt dim-reducing : (%d %d)' % (new_train_data.shape))
    return new_train_data, new_val, new_test, new_unlabel

def mix_pca_gbdt (train_data, train_label, validataion, test, unlabel) :
    """
    """
    if os.path.exists (ROOT + '/data/mix_pca_gbdt') :
        logging.info (ROOT + '/data/mix_pca_gbdt exists!')
        new_train_data, new_val, new_test, new_unlabel = io.grab (ROOT + '/data/mix_pca_gbdt')
    else :
        logging.info ('before mix_pca_gbdt dim-reducing : (%d %d)' % (train_data.shape))
        feature_importance = gbdt_feature_importance (train_data, train_label)
        important_index = np.where (feature_importance > GBDTFEATURETHRESHOLD)[0]
        sorted_index = np.argsort (feature_importance[important_index])[::-1]

        other_index = np.where (feature_importance <= GBDTFEATURETHRESHOLD)[0]
        pca_data = np.vstack ((train_data[:,other_index] , validataion[:,other_index], test[:,other_index], unlabel[:,other_index]))
        pca_data = pca_solver (pca_data)

        new_train_data = np.hstack ((train_data[:,important_index][:,sorted_index], pca_data[:train_data.shape[0],:]))
        new_val = np.hstack ((validataion[:,important_index][:,sorted_index], pca_data[train_data.shape[0]:train_data.shape[0] + validataion.shape[0],:]))
        new_test = np.hstack ((test[:,important_index][:,sorted_index], pca_data[train_data.shape[0]+validataion.shape[0]:-unlabel.shape[0],:]))
        new_unlabel = np.hstack ((unlabel[:,important_index][:,sorted_index], pca_data[-unlabel.shape[0]:,:]))
        logging.info ('after mix_pca_gbdt dim-reducing : (%d %d)' % (new_train_data.shape))
        io.store ([new_train_data, new_val, new_test, new_unlabel], ROOT + '/data/mix_pca_gbdt')
    return new_train_data, new_val, new_test, new_unlabel

def undo (train_data, train_label, val, test, unlabel) :
    """
    nothing to do
    """
    return train_data, val, test, unlabel


