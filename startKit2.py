# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:26:08 2016

@author: benbjo
"""

import pandas as pd
import numpy as np
import math
from sklearn import (preprocessing, model_selection, naive_bayes,
                     linear_model, ensemble)

def load_data():
    x_train = pd.read_csv('../train.csv')
    y_train = x_train['Label']
    relabler = preprocessing.LabelEncoder()
    y_train = pd.Series(relabler.fit_transform(y_train), y_train.index)
    x_train.drop('Label', axis=1, inplace=True)
    x_test = pd.read_csv('../test.csv')
    weight = x_train['Weight']
    EventId_train = x_train['EventId']
    x_train.drop(['EventId', 'Weight'], axis=1, inplace=True)
    EventId_test = x_test['EventId']
    x_test.drop(['EventId'], axis=1, inplace=True)
    
    return x_train, y_train, weight, x_test, EventId_train, EventId_test
    
def calc_ams(y_pred, y_true, weight, wFactor):
    """Calculate AMS 
    """
    idx_s = (y_pred*y_true).astype(bool)
    idx_b = (y_pred*(1-y_true)).astype(bool)
    s = weight[idx_s].sum()*wFactor
    b = weight[idx_b].sum()*wFactor

    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print 'radicand is negative. Exiting'
        exit()
    else:
        return math.sqrt(radicand)
        
def save_submission(EventId, y_pred_proba, y_pred, file_name):
    """y_pred is predicted probabilities
    """
    ranking = np.argsort(y_pred_proba)+1
    classes = np.array(['b', 's'])
    y_pred = classes[y_pred]
    data = {'EventId':EventId,'RankOrder':ranking,'Class':y_pred}

    y_pred = pd.DataFrame(data=data)
    y_pred.to_csv(file_name, index=False, columns=['EventId', 'RankOrder', 
                                                   'Class'])
    
def data_processing(x_train, x_test):
    """Process data
    """
    X = pd.concat([x_train, x_test], ignore_index=True)
    cols = list(X.columns)
    for i, c in enumerate(cols):
        xc = X[c].copy()
        xc_ave = np.mean(xc[xc!=-999])
        X[c][xc==-999] = xc_ave

    X_scale = preprocessing.scale(X)
    X = pd.DataFrame(data=X_scale, columns=cols)

    x_train = X.iloc[:x_train.shape[0],:]
    x_test = X.iloc[x_train.shape[0]:,:]

    return x_train, x_test, X
    
if __name__ == '__main__':
    np.random.seed(0)
    x_train, y_train, weight, x_test, eventid_train, eventid_test = load_data()
#    x_train = x_train+np.random.normal(0.0, 0.0001, x_train.shape)
#    x_test = x_test+np.random.normal(0.0, 0.0001, x_test.shape)
    
    x_train, x_test, X = data_processing(x_train, x_test)

#    relabler = preprocessing.LabelEncoder()
#    sSelector = relabler.fit_transform(y_train).astype(bool)
#    bSelector = (1-sSelector).astype(bool)
#    
#    sumWeights = weight.sum()
#    sumSWeights = weight[sSelector].sum()
#    sumBWeights = weight[bSelector].sum()
#    
#    idx_train, idx_test = \
#    next(model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.1, 
#        random_state=0).split(x_train, y_train))
#    
#    x_train_train = x_train.ix[idx_train]
#    x_train_test = x_train.ix[idx_test]
#    y_train_train = y_train.ix[idx_train]
#    y_train_test = y_train.ix[idx_test] 
#    weight_train = weight.ix[idx_train]
#    weight_test = weight.ix[idx_test]

#    y_pred = np.random.randint(2, size=y_train_test.shape)
#    ams = calc_ams(y_pred, y_train_test, weight_test)
    
    model = naive_bayes.GaussianNB()
    model = linear_model.LogisticRegression()
    model = ensemble.GradientBoostingClassifier(max_depth=9, verbose=10)
    model = ensemble.ExtraTreesClassifier(n_estimators=300, random_state=0,
        n_jobs=7, verbose=10)
    model = ensemble.RandomForestClassifier(n_estimators=500, n_jobs=8, 
        verbose=10)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:,1]
    save_submission(eventid_test, y_pred_proba, y_pred, 
        '../RandomForestClassifier_submission.csv')