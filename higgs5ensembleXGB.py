# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:07:18 2016

@author: lyaa

max_depth = 6, eta = 0.1, n_roungs = 120: 3.63302
"""

from startKit2 import *
import xgboost as xgb
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(0)
    x_train, y_train, weight, x_test, eventid_train, eventid_test = load_data()
#    x_train, x_test, X = data_processing(x_train, x_test)
    
    train_mat = xgb.DMatrix(data=x_train, label=y_train, weight=weight)
    test_mat = xgb.DMatrix(data=x_test)
    
    sum_wpos = weight[y_train==1].sum()
    sum_wneg = weight[y_train==0].sum()
    
    param = {}
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['scale_pos_weight'] = sum_wneg/sum_wpos
    param['eta'] = 0.01
    param['max_depth'] = 9
    param['sub_sample'] = 0.9
    param['eval_metric'] = 'auc'
    param['silent'] = 0
    param['nthread'] = 7
    n_rounds = 3000
    N = 10
    cv = 2
    random_state = 0
    
    scores_auc, x_test_pred_proba, x_train_pred_proba = \
        xgb_ensemble_cv(param, n_rounds, N, x_train, y_train, x_test, weight,
        cv, random_state)
    
    x_test_pred_proba = np.mean(x_test_pred_proba, axis=1)
    x_train_pred_proba = np.mean(x_train_pred_proba, axis=1)
    
    n_steps = 1000
    ams_v = search_th(x_train_pred_proba, y_train, weight, n_steps)
    plt.plot(ams_v)
    
    th_opt = 100.0*np.argmax(ams_v)/n_steps
    
    x_test_pred = cut_ams(x_test_pred_proba, th_opt)
    save_submission(eventid_test, x_test_pred_proba, x_test_pred,
        '../XGB_submission.csv')