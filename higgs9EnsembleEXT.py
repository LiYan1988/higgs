# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:07:18 2016

@author: lyaa

3.63302: max_depth=6, eta=0.1, n_roungs=120
3.71060: 10 repitition 2 fold CV with xgboost 
    max_depth=9, eta=0.01, n_roungs=3000, sub_sample=0.9
3.72067: 10 repitition 3 fold CV with xgboost
"""

from startKit2 import *
import xgboost as xgb
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(0)
    x_train, y_train, weight, x_test, eventid_train, eventid_test = load_data()
    x_train, x_test, X = data_processing(x_train, x_test)
    
    model = ensemble.ExtraTreesClassifier(n_estimators=400, random_state=0,
        n_jobs=7, verbose=0, max_features=30, max_depth=12, 
        min_samples_leaf=100, min_samples_split=100)
    N = 15
    cv = 4
    random_state = 0
    
    scores_auc, x_test_pred_proba, x_train_pred_proba = \
        model_ensemble_cv(model, N, x_train, y_train, x_test, weight,
        cv, random_state, 'EXTEnsemble_test')
    
    x_test_pred_proba = np.mean(x_test_pred_proba, axis=1)
    x_train_pred_proba = np.mean(x_train_pred_proba, axis=1)
    
    n_steps = 1000
    ams_v = search_th(x_train_pred_proba, y_train, weight, n_steps)
    plt.plot(ams_v)
    
    th_opt = 100.0*np.argmax(ams_v)/n_steps
    
    x_test_pred = cut_ams(x_test_pred_proba, th_opt)
    save_submission(eventid_test, x_test_pred_proba, x_test_pred,
        '../CVEnsembleEXT_submission.csv')