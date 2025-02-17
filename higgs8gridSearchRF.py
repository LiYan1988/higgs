# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:05:41 2016

@author: lyaa

EXT: 3.34410
"""

from startKit2 import *
import matplotlib.pyplot as plt
import xgboost as xgb
import os

if __name__ == '__main__':
    np.random.seed(0)
    x_train, y_train, weight, x_test, eventid_train, eventid_test = load_data()
    x_train, x_test, X = data_processing(x_train, x_test)
    
    model = naive_bayes.GaussianNB()
    model = ensemble.RandomForestClassifier(n_estimators=400, random_state=0,
        n_jobs=7, verbose=0, max_features=30, max_depth=12, 
        min_samples_leaf=100, min_samples_split=100)
#    model = ensemble.AdaBoostClassifier(n_estimators=20, learning_rate=0.75,
#        base_estimator=ensemble.ExtraTreesClassifier(n_estimators=400, 
#        max_features=30,  max_depth=12, min_samples_leaf=100, 
#        min_samples_split=100, verbose=1, n_jobs=7))

    params = {}
    params['n_estimators'] = [300, 400, 500, 600, 700, 800]
    params['max_features'] = [20, 25, 30, 35, 40]
    params['max_depth'] = [9, 12, 15, 18]
    params['min_samples_leaf'] = [10, 30, 60, 100]
    params['min_samples_split'] = [10, 30, 60, 100]
    fit_params = {'sample_weight':weight.as_matrix()}
    gridSearch = grid_search.RandomizedSearchCV(model, params, n_iter=100, 
        scoring='roc_auc', cv=3, verbose=10, fit_params=fit_params, 
        random_state=0)
    
    gridSearch.fit(x_train, y_train, )
    
    if not os.path.exists('gridSearch'):
        os.mkdir('gridSearch')
    save_data('gridSearch/randomGridSearchRF.pkl', gridSearch.grid_scores_)