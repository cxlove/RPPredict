#!/usr/bin/env python
# coding=utf-8
"""
> File Name: evaluate.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 四  1/21 17:56:42 2016
"""
import sys
import csv
import logging
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '../..')
from configure import *

def report (scores, n_top = 10) :
    """
    """
    scores = sorted (scores , key = itemgetter (1) , reverse = True)[:n_top]
    for i , score in enumerate (scores) :
        logging.info ("Parameters Rank #%d" % (i))
        print score.parameters
        logging.info ("Validation Score : %.4f , std : %.4f" % (score.mean_validation_score , np.std (score.cv_validation_scores)))
    return scores[0].parameters

def output (uid, predict, filename) :
    """
    """    
    submission = np.array(zip(uid, predict), dtype=[('uid', np.int32),('score', np.float64)])
    predict_file = open(filename, 'wb')
    file_object = csv.writer(predict_file)
    file_object.writerow(['"uid"', '"score"'])
    file_object.writerows(submission)
    predict_file.close()

def get_auc (predict) :
    """
    """
    file = pd.read_csv (ROOT + '/data/val_cv_y.csv')
    label = file.y.values
    logging.info ('The final auc score: %.5f' % roc_auc_score (label, predict))


