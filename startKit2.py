# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:26:08 2016

@author: benbjo
"""

import pandas as pd
import numpy as np
import math
import copy
from sklearn import (preprocessing, model_selection, naive_bayes,
                     linear_model, ensemble, metrics)

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
    y_pred = classes[y_pred.astype(int)]
    data = {'EventId':EventId,'RankOrder':ranking,'Class':y_pred}

    y_pred = pd.DataFrame(data=data)
    y_pred.to_csv(file_name, index=False, columns=['EventId', 'RankOrder', 
                                                   'Class'])
    
def data_processing(x_train, x_test):
    """Process data
    """
    X = pd.concat([x_train, x_test], ignore_index=True)
    X.replace(to_replace=-999, value=np.nan)
    cols = list(X.columns)
    for i, c in enumerate(cols):
        xc = X[c]
        xc_ave = np.mean(xc[xc!=-999])
        X[c].fillna(value=xc_ave, inplace=True)
        if np.median(X[c])/np.std(X[c])>0.25 and (np.max(X[c])-np.min(X[c]))>7:
            X['log'+c] = np.log(1-np.min(X[c])+X[c])

    cols = X.columns
    X_scale = preprocessing.scale(X)
    X = pd.DataFrame(data=X_scale, columns=cols)

    x_train = X.iloc[:x_train.shape[0],:]
    x_test = X.iloc[x_train.shape[0]:,:]

    return x_train, x_test, X
    
def cv_predict_proba(clf, X, y, cv=3, random_state=0):
    """predict probability based on CV
    """
    kf = model_selection.StratifiedKFold(n_splits=cv, shuffle=True,
        random_state=random_state)
    ypred = np.zeros((X.shape[0], len(np.unique(y))))
    for train_index, test_index in kf.split(X, y):
        clf.fit(X.ix[train_index], y[train_index])
        ypred[test_index] = clf.predict_proba(X.ix[test_index])
        
    return ypred
    
def model_ensemble(models, Xtrain, ytrain, Xtest, cv=3, random_state=0, 
                   model_g=None):
    """Ensemble multiple models
    models: list of models 
    model_g: model used to generalize, can be simple average, or linear model
    Xtrain: training set
    ytrain: labels
    Xtest: testing set
    cv: cv folds
    random_state: can be vector
    """
    x_train_pred = np.zeros((Xtrain.shape[0], len(models)))
    x_test_pred = np.zeros((Xtest.shape[0], len(models)))
    scores = np.zeros((len(models), 1))
    for i, model in enumerate(models):
        if type(random_state)==int:
            rs = random_state
        else:
            rs = random_state[i]
        x_train_pred[:, i] = cv_predict_proba(model, Xtrain, ytrain, 
            cv, rs)[:, 1]
        model.fit(Xtrain, ytrain)
        x_test_pred[:, i] = model.predict_proba(Xtest)[:, 1]
        scores[i] = metrics.roc_auc_score(ytrain, x_train_pred[:, i])
        
    if model_g is not None:
        model_g.fit(x_train_pred, ytrain)
        ypred = model_g.predict(x_test_pred) # or predict_proba?
    else:
        ypred = np.mean(x_test_pred)
    
    return ypred, scores
    
def model_ensemble_cv(models, Xtrain, ytrain, Xtest, cv=3, random_state=0):
    """
    """
    x_train_pred_proba = np.zeros((Xtrain.shape[0], len(models)))
    x_test_pred_proba = np.zeros((Xtest.shape[0], len(models)*cv))
    scores_auc = np.zeros((len(models), cv))
    for i, model in enumerate(models):
        if type(random_state)==int:
            rs = np.random.randint(10000)
        else:
            rs = random_state[i]
        kf = model_selection.StratifiedKFold(n_splits=cv, shuffle=True,
            random_state=rs)
        k = 0
        for train_index, test_index in kf.split(Xtrain, ytrain):
            model.fit(Xtrain.iloc[train_index,:], ytrain.iloc[train_index])
            x_train_pred_proba[test_index, i] = \
                model.predict_proba(Xtrain.iloc[test_index, :])[:, 1]
            x_test_pred_proba[:, (i-1)*cv+k] = \
                model.predict_proba(Xtest)[:, 1]
            scores_auc[i, k] = metrics.roc_auc_score(ytrain.iloc[test_index], 
                x_train_pred_proba[test_index, i])
            k = k+1

# Since the sizes of x_train_pred_proba and x_test_pred_proba are not the same,
# this may be a problem
#    if model_g is not None:
#        model_g.fit(x_train_pred_proba, ytrain)
#        ypred = model_g.predict(x_test_pred_proba) # or predict_proba?
#    else:
#        ypred = np.mean(x_test_pred, axis=1)
#    
    return scores_auc, x_test_pred_proba, x_train_pred_proba
    
def cut_ams(y_pred_proba, th=85):
    """Cut probability to signal and backgroud with threshold 
    """
    
    cut = np.percentile(y_pred_proba, th)
    y_pred = y_pred_proba>cut
    
    return y_pred
    
def search_th(y_pred_proba, y_train, weight, n_steps):
    """Brute force search for the best threshold
    """
    ams_v = np.zeros((n_steps, ))
    thresholds = np.arange(0, 100, 100.0/n_steps)
    for i, th in enumerate(thresholds):
        y_pred = cut_ams(y_pred_proba, th=th)
        ams_v[i] = calc_ams(y_pred, y_train, weight, 1)
        
    return ams_v
    
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
    
#    model = naive_bayes.GaussianNB()
#    model = linear_model.LogisticRegression()
#    model = ensemble.GradientBoostingClassifier(max_depth=9, verbose=10)
#    model = ensemble.ExtraTreesClassifier(n_estimators=400, random_state=0,
#        n_jobs=7, verbose=1, max_features=30, max_depth=12, 
#        min_samples_leaf=100, min_samples_split=100)
#    model = ensemble.RandomForestClassifier(n_estimators=500, n_jobs=8, 
#        verbose=10)
#    model.fit(x_train, y_train)
#    y_pred = model.predict(x_test)
#    y_pred_proba = model.predict_proba(x_test)[:,1]
#    save_submission(eventid_test, y_pred_proba, y_pred, 
#        '../RandomForestClassifier_submission.csv')
    N = 15
    models = [copy.deepcopy(model) for i in range(N)]
    np.random.seed(0)
    rsv = np.random.randint(10000000, size=(N,))
    model_g = linear_model.LinearRegression(fit_intercept=False, 
        normalize=False, copy_X=True)
    model_g = None
    y_pred_proba, scores, x_test_pred, x_train_pred = \
        model_ensemble_cv(models, x_train, y_train, x_test, cv=3, 
        random_state=rsv, model_g=model_g)
    y_pred = (y_pred_proba>0.5).astype(int)
    save_submission(eventid_test, y_pred_proba, y_pred, 
        '../EnsembleCVEXT_submission.csv')