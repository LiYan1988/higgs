# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:05:41 2016

@author: lyaa

CV ensemble of EXT: 3.33
"""

from startKit2 import *
import matplotlib.pyplot as plt
import xgboost as xgb

if __name__ == '__main__':
    np.random.seed(0)
    x_train, y_train, weight, x_test, eventid_train, eventid_test = load_data()
    x_train, x_test, X = data_processing(x_train, x_test)
#    x_train = x_train[:10]
#    x_test = x_test[:10]
#    y_train = y_train[:10]
#    weight = weight[:10]
    
    model = naive_bayes.GaussianNB()
    model = ensemble.ExtraTreesClassifier(n_estimators=600, random_state=0,
        n_jobs=7, verbose=1, max_features=42, max_depth=12, 
        min_samples_leaf=100, min_samples_split=100)
#    model = ensemble.AdaBoostClassifier(n_estimators=20, learning_rate=0.75,
#        base_estimator=ensemble.ExtraTreesClassifier(n_estimators=400, 
#        max_features=30,  max_depth=12, min_samples_leaf=100, 
#        min_samples_split=100, verbose=1, n_jobs=7))

    N = 10
    models = [copy.deepcopy(model) for i in range(N)]
    scores, x_test_pred_proba, x_train_pred_proba = \
        model_ensemble_cv(models, x_train, y_train, x_test, cv=2, 
        random_state=0)
    x_train_pred = np.mean(x_train_pred_proba, axis=1)
        
    n_steps = 1000
    ams_v = search_th(x_train_pred, y_train, weight, n_steps)
    plt.plot(ams_v)
    
    th_opt = 100.0*np.argmax(ams_v)/n_steps
    
    y_test_pred_proba = np.mean(x_test_pred_proba, axis=1)
    y_test_pred = cut_ams(y_test_pred_proba, th_opt)
    save_submission(eventid_test, y_test_pred_proba, y_test_pred,
        '../ensembleCVEXT_submission.csv')