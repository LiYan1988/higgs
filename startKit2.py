# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:26:08 2016

@author: benbjo
"""

import pandas as pd
import numpy as np

def load_data():
    x_train = pd.read_csv('train.csv', compression='gzip', index_col=0)
    y_train = x_train['Label']
    x_train.drop('Label', axis=1, inplace=True)
    x_test = pd.read_csv('test.csv', compression='gzip', index_col=0)
    weight = x_train['Weight']
    x_train.drop(['EventId', 'Weight'], axis=1, inplace=True)
    x_test.drop(['EventId'], axis=1, inplace=True)
    
    return x_train, y_train, weight, x_test
    

    
if __name__ == '__main__':
    x_train, y_train, weight, x_test = load_data()
    