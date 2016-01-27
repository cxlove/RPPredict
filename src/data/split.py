#!/usr/bin/env python
# coding=utf-8
"""
> File Name: split.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 一  1/18 10:28:20 2016
"""

import sys
import random 

sys.path.append('../..')
from configure import *

def split_train_test () :
    x = open (ROOT + '/data/train_x.csv').readlines ()
    y = open (ROOT + '/data/train_y.csv').readlines ()

    train_x = open (ROOT + '/data/train_cv_x.csv', 'w')
    train_y = open (ROOT + '/data/train_cv_y.csv', 'w')
    test_x = open (ROOT + '/data/val_cv_x.csv', 'w')
    test_y = open (ROOT + '/data/val_cv_y.csv', 'w')

    train_x.write (x[0])
    train_y.write (y[0])
    test_x.write (x[0])
    test_y.write (y[0])

    for i in xrange (1, len (x)) :
        val = random.randint (1, TESTDATA + TRAINDATA)
        if val <= TRAINDATA :
            train_x.write (x[i])
            train_y.write (y[i])
        else :
            test_x.write (x[i])
            test_y.write (y[i])

def split_pos_neg () :
    x = open (ROOT + '/data/train_x.csv').readlines ()
    y = open (ROOT + '/data/train_y.csv').readlines ()

    train_x = open (ROOT + '/data/train_cv_x.csv', 'w')
    train_y = open (ROOT + '/data/train_cv_y.csv', 'w')
    test_x = open (ROOT + '/data/val_cv_x.csv', 'w')
    test_y = open (ROOT + '/data/val_cv_y.csv', 'w')

    train_x.write (x[0])
    train_y.write (y[0])
    test_x.write (x[0])
    test_y.write (y[0])

    pos = []
    neg = []

    for i in xrange (1, len (x)) :
        line = map (float, y[i].split (',')) 
    

        val = random.randint (1, TESTDATA + TRAINDATA)
        if val <= TRAINDATA :
            if line[1] == 1 : pos.append ([x[i], y[i]])
            else : neg.append ([x[i], y[i]])
        else :
            test_x.write (x[i])
            test_y.write (y[i])

    print len (neg) , len (pos)

    random.shuffle(pos)
    random.shuffle(neg)
    for i in xrange (len (neg)) :
        train_x.write (neg[i][0])
        train_y.write (neg[i][1])
        for j in xrange (4) :
            train_x.write (pos[i * 4 + j][0])
            train_y.write (pos[i * 4 + j][1])

if __name__ == '__main__' :
    split_pos_neg ()
    



