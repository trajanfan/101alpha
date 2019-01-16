# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:43:55 2018

@author: Trajan
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from config import config
from Parameters import param_space
from Functions import quadratic_weighted_kappa

def model_set(param):
    
    kappa_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
    
    if config.cv:
        n_fold = config.n_folds
        fold_path = "Fold%d"
    else:
        n_fold = 1
        fold_path = ''
        
    for run in range(1,config.n_runs+1):
        for fold in range(1,n_fold+1):
            
            X_train = pd.read_hdf(config.train_label_path + fold_path%fold)
            labels_train = pd.read_csv(config.train_label_path + fold_path%fold)
            X_valid = pd.read_hdf(config.valid_label_path + fold_path%fold)
            labels_valid = pd.read_csv(config.valid_label_path + fold_path%fold)
            numValid = len(labels_valid)
            
            dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
            dtrain_base = xgb.DMatrix(X_train, label=labels_train)
            num_round = param['num_round']
            preds_bagging = np.zeros((numValid, config.bagging_size), dtype=float)
            
            for n in range(config.bagging_size):
                param = param_space[n]
                if param["task"] in ["regression"]:
                    ## regression & pairwise ranking with xgboost
                    bst = xgb.train(param, dtrain_base, num_round)
                    pred = bst.predict(dvalid_base)
            
                elif param["task"] in ["softmax"]:
                    ## softmax regression with xgboost
                    bst = xgb.train(param, dtrain_base, num_round)
                    pred = bst.predict(dvalid_base)
                    w = np.asarray(range(1,config.numOfClass+1))
                    pred = pred * w[np.newaxis,:]
                    pred = np.sum(pred, axis=1)
            
                elif param['task'] == "reg_skl_svr":
                    ## regression with sklearn support vector regression
                    X_train, X_valid = X_train.toarray(), X_valid.toarray()
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_valid = scaler.transform(X_valid)
                    svr = SVR(C=param['C'], gamma=param['gamma'], kernel=param['kernel'])
                    svr.fit(X_train, labels_train+1)
                    pred = svr.predict(X_valid)
            
                elif param['task'] == "reg_skl_ridge":
                    ## regression with sklearn ridge regression
                    ridge = Ridge(alpha=param["alpha"], normalize=True)
                    ridge.fit(X_train, labels_train+1)
                    pred = ridge.predict(X_valid)
            
                elif param['task'] == "reg_skl_lasso":
                    ## regression with sklearn lasso
                    lasso = Lasso(alpha=param["alpha"], normalize=True)
                    lasso.fit(X_train, labels_train+1)
                    pred = lasso.predict(X_valid)
                
                
                ## weighted averageing over different models
                pred_valid = pred
                ## this bagging iteration
                preds_bagging[:,n] = pred_valid
                pred_raw = np.mean(preds_bagging[:,:(n+1)], axis=1)
                pred_rank = pred_raw.argsort().argsort()
                pred_score = np.sign(pred_rank)
                kappa_valid = quadratic_weighted_kappa(pred_score, labels_valid)
                if (n+1) != config.bagging_size:
                    print("              {:>3}   {:>3}   {:>3}".format(
                            run, fold, n+1, np.round(kappa_valid,6)))
                else:
                    print("                    {:>3}       {:>3}      {:>3}".format(
                            run, fold, n+1, np.round(kappa_valid,6)))
    
            kappa_cv[run-1,fold-1] = kappa_valid
    
    
    kappa_cv_mean = np.mean(kappa_cv)
    kappa_cv_std = np.std(kappa_cv)
    return -kappa_cv_mean, kappa_cv_std
        
if __name__ == "__main__":
    kappa_cv_mean, kappa_cv_std = model_set(param_space)
    print('mean'%d)