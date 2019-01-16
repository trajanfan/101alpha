# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:41:37 2018

@author: Trajan
"""

import os
import numpy as np


class config:
    def __init__(self,count_feat_transform=np.sqrt):
    
        
        self.bagging_size = 1
        self.numOfClass = 2

        ## CV params
        self.cv = False
        self.n_runs = 3
        self.n_folds = 3

        ## path
        self.rootPath = './'
        self.train_feature_path = self.rootPath+'data/train.h5'
        self.train_label_path = self.rootPath+'data/train_labels.csv'
        self.valid_feature_path = self.rootPath+'data/valid.h5'
        self.valid_label_path = self.rootPath+'data/valid_labels.csv'

        ## transform for count features
        self.count_feat_transform = count_feat_transform

        ## creat folder for each run and fold
        if self.cv:
            for run in range(1,self.n_runs+1):
                for fold in range(1,self.n_folds+1):
                    path = self.rootPath + "Fold%d" %fold
                    if not os.path.exists(path):
                        os.makedirs(path)


## initialize a param config					
config = config()