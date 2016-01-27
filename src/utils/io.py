#!/usr/bin/env python
# coding=utf-8
"""
> File Name: io.py
> Author: cxlove
> Mail: cxlove321@gmail.com
> Created Time: 四  1/21 17:53:11 2016
"""

import pickle 

def store (input, filename) :
    """
    store input into filename used pickle.dump
    """
    cout = open (filename, 'w')
    pickle.dump (input, cout)
    cout.close ()

def grab (filename) :
    """
    load data from filename used pickle.load
    """
    cin = open (filename, 'r')
    return pickle.load (cin)

