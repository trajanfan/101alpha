# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:10:55 2019

@author: Trajan
"""


## xgboost
xgb_random_seed = 2015
xgb_nthread = 2
xgb_dmatrix_silent = True

## sklearn
skl_random_seed = 2015


param_space_reg_xgb_linear = {
    'task': 'regression',
    'booster': 'gblinear',
    'objective': 'reg:linear',
    'nthread': xgb_nthread,
    'silent' : 1,
    'seed': xgb_random_seed,
}
param_space_clf_xgb_linear = {
    'task': 'softmax',
    'booster': 'gblinear',
    'objective': 'multi:softprob',
    'num_class': 2,
    'nthread': xgb_nthread,
    'silent' : 1,
    'seed': xgb_random_seed,
}
param_space_reg_skl_svr = {
    'task': 'reg_skl_svr',
    'C': 0,
    'gamma': -4,
    'kernal': 'rbf'
}
param_space_reg_skl_ridge = {
    'task': 'reg_skl_ridge',
    'alpha': 1,
    'random_state': skl_random_seed,
}
param_space_reg_skl_lasso = {
    'task': 'reg_skl_lasso',
    'alpha': -2,
    'random_state': skl_random_seed,
}

param_space = {
    3:param_space_reg_xgb_linear,
    5:param_space_clf_xgb_linear,
    2:param_space_reg_skl_svr,
    4:param_space_reg_skl_ridge,
    1:param_space_reg_skl_lasso
}