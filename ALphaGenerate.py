# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:43:02 2019

@author: Trajan
"""

import pandas as pd
import numpy as np
import sys
import pickle

import pandas_market_calendars as mcal
from AlphaFactory import AlphaFactory
import time
        

class Alpha_Calculation():
   
    def __init__(self, DATA):
        self.Parameter = {'alpha_1': {'std_window':20,'argmax_window':5},                     
             'alpha_2': {'delta_step':2,'corr_window':6,'alpha_sign':-1},        
             'alpha_3': {'alpha_sign':-1, 'corr_window':10},       
             'alpha_4': {'alpha_sign':-1, 'tsrank_window':9},                    
             'alpha_5': {'alpha_sign':-1, 'mean_window':10},                   
             'alpha_6': {'alpha_sign':-1, 'corr_window':10},                    
             'alpha_7': {'alpha_sign':-1, 'delta_step':7, 'tsrank_window':60},
             'alpha_8': {'alpha_sign':-1, 'lookback_window':5, 'delay_step':10},
             'alpha_9': {'ts_window':5, 'delta_step':1},
             'alpha_10': {'ts_window':4, 'delta_step':1},                        
             'alpha_11': {'lookback_window':3},                                                         
             'alpha_12': {'delta_step':1, 'alpha_sign':-1},                                                     
             'alpha_13': {'alpha_sign':-1, 'cov_window':5},                                                     
             'alpha_14': {'alpha_sign':-1, 'delta_step':3, 'corr_window':5},                                    
             'alpha_15': {'alpha_sign':-1, 'corr_window':3, 'sum_window':3},                                    
             'alpha_16': {'alpha_sign':-1, 'cov_window':5},                                                     
             'alpha_17': {'alpha_sign':-1, 'delta_step':1, 'tsrank_close_window':10, 'tsrank_volume_window':5}, 
             'alpha_18': {'alpha_sign':-1, 'std_window':5, 'corr_window':10},                                   
             'alpha_19': {'alpha_sign':-1, 'delta_step':7, 'sum_window':250},                                   
             'alpha_20': {'alpha_sign':-1, 'lookback_window':1},
             'alpha_21': {'alpha_sign':-1, 'lookback_window':8, 'current_window':2},
             'alpha_22': {'alpha_sign':-1, 'corr_window':5, 'delta_step':5, 'std_window':20},
             'alpha_23': {'alpha_sign':-1, 'mean_window':20, 'delta_step':2},
             'alpha_24': {'alpha_sign':-1, 'change_window':100, 'mean_window':100, 'pct_change':0.05, 'delta_step':3, 'min_window':100},
             'alpha_25': {'alpha_sign':-1},
             'alpha_26': {'alpha_sign':-1, 'tsrank_window':5, 'corr_window':5, 'tsmax_window':3},
             'alpha_27': {'rank_pct':0.5, 'corr_window':6, 'mean_window':2, 'alpha_sign':-1},
             'alpha_28': {'corr_window':5},
             'alpha_29': {'delta_step':5, 'tsmin_window':2, 'sum_window':1, 'prod_window':1, 'min_window':5, 'delay_step':6, 'tsrank_window':5, 'alpha_sign':-1},
             'alpha_30': {'delta_step':1, 'delay_steps':2, 'sum_window1':5, 'sum_window2':20},
             'alpha_31': {'alpha_sign':-1, 'delta_step':10 , 'decay_window':10, 'delta_step2':3, 'corr_window':12},
             'alpha_32': {'mean_window':7, 'scale_para':20, 'delay_step':5, 'corr_window':230},
             'alpha_33': {'alpha_sign':-1, 'pow_level':1},
             'alpha_34': {'std_window1':2, 'std_window2':5, 'delta_step':1},
             'alpha_36': {'coef1':2.21, 'delay_step':1, 'corr_window':15,'coef2':0.7, 
                           'coef3':0.73, 'alpha_sign':-1, 'delay_step2':6, 'tsrank_window':5,
                           'coef4':1, 'corr_window2':6,'coef5':0.6, 'mean_window':200},
             'alpha_35': {'tsrank_window1':32, 'tsrank_window2':16, 'tsrank_window3':32},
             'alpha_37': {'delay_step':1, 'corr_window':200},
             'alpha_38': {'alpha_sign':-1, 'tsrank_window':10},
             'alpha_39': {'alpha_sign':-1, 'delta_step':7, 'decay_window':9, 'sum_window':250},
             'alpha_40': {'alpha_sign':-1, 'lookback_window':10},
             'alpha_43': {'vol_window':20,'close_window':8},
             'alpha_44': {'corr_window':5},
             'alpha_45': {'delay_window':5,'sum_window':20,'corr_window':2},
             'alpha_46': {'lookback_window':10},
             'alpha_47': {'vol_window':20,'sum_window':5,'delay_window':5},
             'alpha_48': {'short_window':1,'long_window':250},
             'alpha_49': {'lookback_window':10},
             'alpha_50': {'lookback_window':5},
             'alpha_51': {'lookback_window':10},
             'alpha_52': {'lookback_window':5,'long_sum_window':240,'short_sum_window':20},
             'alpha_53': {'lookback_window':9},
             'alpha_54': {'price_power':5},
             'alpha_55': {'minmax_window':12,'corr_window':6},
             'alpha_56': {'long_window':10,'mid_window':3,'short_window':2,'cap_window':20},
             'alpha_57': {'lookback_window':30,'decay_window':2},
             'alpha_58': {'corr_window':3,'decay_window':7,'rank_window':5},
             'alpha_59': {'factor':0.728317,'corr_window':4,'decay_window':16,'rank_window':8},
             'alpha_60': {'lookback_window':10},
             'alpha_61': {'minwindow':16,'corrwindow':17},
             'alpha_62': {'sumwindow':22,'corrwindow':9},
             'alpha_63': {'deltawindow':2,'decaywindow':8,'weight':0.318108,'sumwindow':37,'corrwindow':13,'decaywindow2':12},
             'alpha_64': {'weight':0.178404,'sumwindow':12,'corrwindow':16,'weight2':0.178404,'deltawindow':3},
             'alpha_65': {'weight':0.00817205,'sumwindow':8,'corrwindow':6,'minwindow':13},
             'alpha_66': {'weight':0.96633,'deltawindow':3,'decaywindow1':7,'decaywindow2':11,'rankwindow':6},
             'alpha_67': {'rankwindow':2,'corrwindow':6},
             'alpha_68': {'corrwindow':8,'rankwindow':13,'weight':0.518371,'deltawindow':1},
             'alpha_69': {'weight':0.490655,'deltawindow':2,'maxwindow':4,'corrwindow':4,'rankwindow':9},
             'alpha_70': {'deltawindow':1,'corrwindow':17,'rankwindow':17},
             'alpha_71': {'rankwindow1':3,'rankwindow2':12,'rankwindow3':15,'rankwindow4':4,'corrwindow':18,'decaywindow1':4,'decaywindow2':16},
             'alpha_72': {'corrwindow1':8,'corrwindow2':6,'decaywindow1':10,'decaywindow2':2,'rankwindow1':3,'rankwindow2':18},
             'alpha_73': {'weight':0.147155,'deltawindow1':4,'decaywindow1':2,'deltawindow2':2,'decaywindow2':3,'rankwindow':16},
             'alpha_74': {'weight':0.0261661,'corrwindow1':15,'corrwindow2':11,'sumwindow':37},
             'alpha_75': {'corrwindow1':4,'corrwindow2':12},
             'alpha_76': {'deltawindow':1,'decaywindow1':11,'corrwindow':8,'rankwindow1':19,'decaywindow2':17,'rankwindow2':19},
             'alpha_77': {'decaywindow1':20,'decaywindow2':5,'corrwindow':3},
             'alpha_78': {'weight':0.352233,'sumwindow':19,'corrwindow1':6,'corrwindow2':5},
             'alpha_79': {'weight':0.60733,'deltawindow':1,'rankwindow1':3,'rankwindow2':9,'corrwindow':14},
             'alpha_80': {'weight':0.868128,'deltawindow':4,'corrwindow':5,'rankwindow':5},
             'alpha_81': {'sum_window':49.6054,'corr_volume_window':8.47743,'rank_power':4,'prod_window':14.9655,'corr_rankvolume_window':5.07914,'alpha_sign':-1},
             'alpha_82': {'delta_step':1.46063,'ldecay_window':14.8717,'open_scale':0.634196,'corr_window':17.4842,'ldecay_2_window':6.92131,'tsrank_window':13.4283,'alpha_sign':-1},
             'alpha_83': {'sum_window':5,'delay_step':2},
             'alpha_84': {'tsmax_window':15.3217,'tsrank_window':20.7127,'delta_step':4.96796},
             'alpha_85': {'high_pct':0.876703,'corr_window_1':9.61331,'tsrank_highlow_window':3.70596,'tsrank_volume_window':10.1595,'corr_window_2':7.11408},
             'alpha_86': {'sum_window':14.7444,'corr_window':6.00049,'tsrank_window':20.4195,'alpha_sign':-1},
             'alpha_87': {'adv_window':81,'close_pct':0.369701,'delta_step':1.91233,'ldecay_window_1':2.65461,'corr_window':13.4132,'ldecay_window_2':4.89768,'tsrank_window':14.4535,'alpha_sign':-1},
             'alpha_88': {'ldecay_window_1':8.06882,'tsrank_window_1':8.44728,'tsrank_window_2':20.6966,'corr_window':8.01266,'ldecay_window_2':6.65053,'tsrank_window_3':2.61957},
             'alpha_89': {'low_pct':0.967285,'corr_window':6.94279,'ldecay_window':5.51607,'tsrank_window':3.79744,'delta_step':3.48158,'ldecay_window_2':10.1466,'tsrank_window_2':15.3012},
             'alpha_90': {'tsmax_window':4.66719,'corr_window':5.38375,'tsrank_window':3.21856,'alpha_sign':-1},
             'alpha_91': {'corr_window':9.74928,'ldecay_window':16.398,'ldecay_window_2':3.83219,'tsrank_window':4.8667,'corr_window_2':4.01303,'ldecay_window_3':2.6809,'alpha_sign':-1},
             'alpha_92': {'ldecay_window_1':14.7221,'tsrank_window_1':18.8683,'corr_window':7.58555,'ldecay_window_2':6.94024,'tsrank_window_2':6.80584},
             'alpha_93': {'adv_window':81,'corr_window':17.4193,'ldecay_window_1':19.848,'tsrank_window':7.54455,'close_pct':0.524434,'delta_step': 2.77377,'ldecay_window_2':16.2664},
             'alpha_94': {'tsmin_window':11.5783,'tsrank_window_1':19.6462,'tsrank_window_2':4.02992,'corr_window':18.0926,'tsrank_window_3':2.70756,'alpha_sign':-1},
             'alpha_95': {'tsmin_window':12.4105,'sum_window':19.1351,'corr_window':12.8742,'power_times':5,'tsrank_window': 11.7584,'alpha_sign':-1},             
             'alpha_96': {'corr_window_1': 3.83878, 'ldecay_window_1': 4.16783, 'tsrank_window_1': 8.38151, 'tsrank_window_2': 7.45404, 'tsrank_window_3': 4.13242, 'corr_window_2': 3.65459, 'tsargmax_window': 12.6556, 'ldecay_window_2': 14.0365, 'tsrank_window_4': 13.4143, 'alpha_sign': -1},            
             'alpha_97': {'low_pct':0.721001,'delta_window':3.3705,'ldecay_window_1':20.4523,'tsrank_window_1':7.87871,'tsrank_window_2':17.255,'corr_window':4.97547,'tsrank_window_3': 18.5925,'ldecay_window_2':15.7152,'tsrank_window_4':6.71659,'alpha_sign':-1},             
             'alpha_98': {'adv_window':5,'sum_window':26.4719,'corr_window':4.58418,'ldecay_window_1':7.18088,'corr_window_2':20.8187,'tsargmin_window':8.62571,'tsrank_window_1':6.95668,'ldecay_window_2':8.07206},             
             'alpha_99': {'sum_window':19.8975,'corr_window_1':8.8136,'corr_window_2':6.28259,'alpha_sign':-1},             
             'alpha_100': {'corr_window':5,'tsargmin_window':30,'scale_1':1.5,'scale_2':1,'threshold':0},
             'alpha_101': {'threshold':0.001}}
        self.DATA = DATA
        
    def Features(self, start, end, store):
        for N in range(start, end):
            try:
                start = time.time()
                alphaName = 'Alpha'+str(N)
                
                if alphaName in store:
                    print (alphaName + ' exits; start loading from file...')
                    test = store.get(alphaName)
                else:
                    print (alphaName + ' does not exit; start calculating...')
                    test = AlphaFactory.CalculateAlphas(DATA = self.DATA, Alpha = N, PARAMETER = self.Parameter)
                    store.put(alphaName, test)
                    
                elapsed = time.time() - start
                print("Alpha%d is finished. Time used: %.2f seconds"%(N,elapsed))        
                    
            except Exception as e:
                print(e.args)
                pass
    
    def Labels(self, start, end, store):
        for N in range(start, end):
            try:                    
                DATA = self.DATA
                df_return = DATA['return'].shift(-1)
                median_list = df_return.median(axis=1,skipna=True)
                for i in SYMBOLS:
                    df_return[i] = np.sign(df_return[i]-median_list)
                df_return.to_csv(store)
                    
            except Exception as e:
                print(e.args)
                pass
            
if __name__ == "__main__":
    
    #All data
    rootPath=''
    file = open(rootPath+'data/all_data.pkl','rb')
    DATA = pickle.load(file)
    file.close()
    
    #Valid S&P 500stocks
    names=['Symbol','Name','Sector']
    df = pd.read_csv(rootPath+'data/sp500.csv', header=0, names=names)
    SYMBOLS = pd.Index(df['Symbol']).intersection(pd.Index(DATA['open'].columns)).values
    
    #Divide data set
    nyse = mcal.get_calendar('NYSE')
    train_DATES = pd.Series(nyse.valid_days(start_date='2011-01-01', end_date='2012-12-31').date)
    valid_DATES = pd.Series(nyse.valid_days(start_date='2013-01-01', end_date='2014-12-31').date)
    
    train_DATA = {}
    valid_DATA = {}
    for key in DATA.keys():
        if key!='ind': 
            train_DATA[key] = DATA[key].loc[train_DATES]
            train_DATA[key] = train_DATA[key][SYMBOLS]
            valid_DATA[key] = DATA[key].loc[valid_DATES]
            valid_DATA[key] = valid_DATA[key][SYMBOLS]
        else:
            train_DATA[key] = DATA[key].loc[SYMBOLS]
            valid_DATA[key] = DATA[key].loc[SYMBOLS]
    
    sys.path.append('')
       
    
    #Save results
    store_f = pd.HDFStore(rootPath+'data/train.h5')
    store_l = rootPath+'data/train_labels.csv'
    ts = Alpha_Calculation(train_DATA)
    #ts.Features(1, 102, store_f)
    ts.Labels(1, 102, store_l)
    store_f.close()
    store_f = pd.HDFStore(rootPath+'data/valid.h5')
    store_l = rootPath+'data/valid_labels.csv'
    vs = Alpha_Calculation(valid_DATA)
    #vs.Features(1, 102, store_f)
    vs.Labels(1, 102, store_l)
    store_f.close()