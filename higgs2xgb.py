# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:07:18 2016

@author: lyaa
"""

from startKit2 import *
import xgboost as xgb

if __name__ == '__main__':
    np.random.seed(0)
    x_train, y_train, weight, x_test, eventid_train, eventid_test = load_data()
    x_train, x_test, X = data_processing(x_train, x_test)
    
    train_mat = xgb.DMatrix(data=x_train, label=y_train, weight=weight)
    
    sum_wpos = weight[y_train==1].sum()
    sum_wneg = weight[y_train==0].sum()
    
    param = {}
    param['objective'] = 'binary:logistic'
    # scale weight of positive examples
    param['scale_pos_weight'] = sum_wneg/sum_wpos
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['eval_metric'] = 'auc'
    param['silent'] = 0
    param['nthread'] = 7

    n_rounds = 300
    bst = xgb.train(param, train_mat)
    y_train_pred_proba = bst.predict(train_mat)
    
    n_steps = 1000
    ams_v = search_th(y_pred_proba, y_train, weight, n_steps)
    plt.plot(ams_v)
    
    th_opt = 100.0*np.argmax(ams_v)/n_steps
    
    y_test_pred_proba = model.predict_proba(x_test)[:,1]
    y_test_pred = cut_ams(y_test_pred_proba, th_opt)
    save_submission(eventid_test, y_test_pred_proba, y_test_pred,
        '../XGB_submission.csv')