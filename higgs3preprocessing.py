# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:26:45 2016

@author: lyaa
"""

from startKit2 import *
import xgboost as xgb

x_train, y_train, weight, x_test, eventid_train, eventid_test = load_data()
x_train, x_test, X = data_processing(x_train, x_test)
cols = x_train.columns

#%%
q1_v = np.zeros(cols.shape)
q2_v = np.zeros(cols.shape)
q3_v = np.zeros(cols.shape)
r_v = np.zeros(cols.shape)
for i, c in enumerate(cols):
    q1_v[i] = np.percentile(X[c], 25)
    q2_v[i] = np.percentile(X[c], 50)
#    q3_v[i] = np.percentile(X[c], 75)
    q3_v[i] = np.std(X[c])
#    r_v[i] = 0.5*abs((q1_v[i]+q3_v[i])/q2_v[i])
    r_v[i] = q2_v[i]/q3_v[i]