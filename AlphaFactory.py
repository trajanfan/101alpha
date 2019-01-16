#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:43:55 2018

@author: Trajan
"""

import pandas as pd
import numpy as np

class AlphaFactory(object):
    '''CloudQuant Alpha/Features generation class'''
    def __init__(self):
        pass
    
    @staticmethod
    def CalculateAlphas( *args, **kwargs):
        '''
        Input: 
            DATA: input data
            Alpha: the choice of alpha selected to calculate.
        Return:
            df_return: pands DataFrame with desired features.
        '''

        #Prepare data (Global)
        Alpha = kwargs.get('Alpha')
        DATA = kwargs.get('DATA')
        PARAMETER = kwargs.get('PARAMETER')
        
        df_open = DATA['open']
        df_high = DATA['high']
        df_low = DATA['low']
        df_close = DATA['close']
        df_volume = DATA['volume']
        df_vwap = DATA['vwap']
        df_return = DATA['return']
        df_nan = DATA['nan']
        df_ind = DATA['ind']
        
        #Prepare base functions
        def df_adv(window=10):
            return df_volume.rolling(window=window).mean()
        
        def df_cap(window=10):
            df = df_vwap * df_volume
            return df.rolling(window=window).mean()
        
        def product(df_x=None, window=None):
            return df_x.rolling(window=window).apply(func=lambda x: x.prod())

        def rank(df):
            return df.rank(axis=1, method='min', na_option='keep', ascending=False, pct=True)

        def delta(df, days=1):
            return df.diff(days)

        def delay(df, days=1):
            return df.shift(days)

        def correlation(df1, df2, window=10):
            ret = df1.rolling(window=window).corr(other=df2)
            ret[np.isinf(ret)]=0
            return ret
        
        def covariance(df1, df2, window=10):
            return df1.rolling(window=window).cov(other=df2)

        def vec_scale(ts, norm=1):
            abssum = np.abs(ts).sum()
            if abssum > 0:
                return ts / abssum * norm
            else:
                return ts
            
        def scale(df, norm=1):
            return df.apply(vec_scale, axis=1, norm=norm)

        def ts_rank(df, window=10):
            rankDf = pd.DataFrame(index=df.columns)
            for i in df.index:
                temp = df[:i][-window:].rank(axis=0, method='min', na_option='keep', ascending=False, pct=True).iloc[-1]
                rankDf[i] = temp
            return rankDf.T
        
        def self_rank(series):
            rank_series=(pd.Series.rank(series)-1)/len(series)
            return rank_series.iloc[-1]
        def ts_rank_one(series,d):
            return series.rolling(window=d).apply(lambda x: self_rank(pd.Series(x)))
        def ts_rank2(dataframe,d):
            return dataframe.apply(lambda x:ts_rank_one(x,d))

        def ts_mean(df, window=10):
            return df.rolling(window=window).mean()

        def ts_min(df, window=10):
            return df.rolling(window=window).min()

        def ts_max(df, window=10):
            return df.rolling(window=window).max()

        def ts_argmax(df, window=10):
            return df.rolling(window=window).apply(np.argmax)   

        def ts_argmin(df, window=10):
            return df.rolling(window=window).apply(np.argmin)

        def ts_sum(df, window=10):
            return df.rolling(window=window).sum()

        def ts_product(df, window=10):
            return df.rolling(window=window).apply(np.prod)

        def ts_stddev(df, window=10):
            return df.rolling(window=window).std()
        
        def decay_linear(data, window=10):
            result = pd.DataFrame(
                np.nan, index=data.index, columns=data.columns)
            weight = np.arange(window) + 1.
            weight = weight / weight.sum()
            for i in range(window, data.shape[0] + 1):
                result.iloc[i - 1, :] = data.iloc[i - window:i].T.dot(weight)
            return result

        def decay_linear_one(dataframe,d):
            sum=0
            for i in range(1,d+1):
                sum+=(d+1-i)*dataframe.iloc[-i]
            return 2*sum/((1+d)*d)
        def decay_linear2(dataframe,d):
            return dataframe.rolling(window=d).apply(lambda x:decay_linear_one(pd.DataFrame(x),d))

        def IndNeutralize(df, ind):
            return (df.T.groupby(ind).apply(lambda x: x - np.nanmean(x,axis=0))).T

        def SignedPower(df, a=1):
            return df.pow(a)
        
        def Max(df1, df2):
            res = df1.copy() * np.nan
            mask = df1 > df2
            mask2 = df1 <= df2
            res[mask] = df1[mask]
            res[mask2] = df2[mask2]
            return res

        def Min(df1, df2):
            res = df1.copy() * np.nan
            mask = df1 <= df2
            mask2 = df1 > df2
            res[mask] = df1[mask]
            res[mask2] = df2[mask2]
            return res
            
        def min(df1,df2):
            fac=df1
            fac[df1>df2]=df2
            return fac
            
        def max(df1,df2):
            fac=df1
            fac[df1<df2]=df2
            return fac

        def Log(df):
            ret = df.copy()
            ret[df==0] = -np.inf
            ret[df>0] = np.log(df[df>0])
            ret[df<0] = np.nan
            return ret

        def sign(df):
            return df.applymap(lambda x:np.sign(x))
        
        def nanint(x):
            try:
                return int(x)
            except:
                return np.NaN
        def Df_Int(df):
            return df.applymap(lambda x:nanint(x))

        #Now begin the Alpha functions
        def Alpha1():
            '''
            Formula: rank(Ts_ArgMax(stddev(Return,Std_window),Argmax_window)))-0.5
            Parameter: std_window = 20, argmax_window = 5
            Explanation: Try to long the most recently volatile stock
            '''
            para1 = PARAMETER['alpha_1']['std_window']
            para2 = PARAMETER['alpha_1']['argmax_window']
            return rank(ts_max(ts_stddev(df_return, para1), para2)) - 0.5

        def Alpha2():
            '''
            Formula: alpha_sign * correlation(rank(delta(log(volume),delta_step)),rank(((close-open)/open)),corr_window)
            Parameter: delta_step = 2, corr_window = 6, alpha_sign = -1
            Explanation: Long the stock with different change directions in ranks of volume and price pct change daily
            '''
            para1 = PARAMETER['alpha_2']['delta_step']
            para2 = PARAMETER['alpha_2']['corr_window']
            para3 = PARAMETER['alpha_2']['alpha_sign']
            temp = (df_close - df_open) / df_open
            return para3 * correlation(rank(delta(Log(df_volume), para1)),rank(temp),para2)

        def Alpha3():
            '''
            Formula: alpha_sign * correlation(rank(open),rank(volume),corr_window)
            Parameter: alpha_sign = -1, corr_window = 10
            Explanation: long stocks with negative coorelation between rank of volume and open price
            '''
            para1 = PARAMETER['alpha_3']['alpha_sign']
            para2 = PARAMETER['alpha_3']['corr_window']
            return para1 * correlation(rank(df_open),rank(df_volume),para2)

        def Alpha4():
            '''
            Formula: alpha_sign * Ts_rank(rank(low),tsrank_window)
            Parameter: alpha_sign = -1, tsrank_window = 9
            Explanation:  ALERT! THIS FORMULA MIGHT BE WRONG,ALPHA WILL ALWAYS BE NEGATIVE
            '''
            para1 = PARAMETER['alpha_4']['alpha_sign']
            para2 = PARAMETER['alpha_4']['tsrank_window']
            return para1 * ts_rank(rank(df_low),window = para2)

        def Alpha5():
            '''
            Formula: rank((open-(mean(vwap,mean_window)))*(alpha_sign*abs(rank((close-vwap)))))
            Parameter: alpha_sign = -1, mean_window = 10
            Explanation: ALERT! FORMULA HAS REDUNDENT FUNCTION ABS, COULD BE WRONG
            '''
            para1 = PARAMETER['alpha_5']['alpha_sign']
            para2 = PARAMETER['alpha_5']['mean_window']
            return rank((df_open-ts_mean(df_vwap,window = para2))*para1*rank(df_close-df_vwap))
        
        def Alpha6():
            '''
            Formula: alpha_sign * correlation(open,volume,corr_window)
            Parameter: alpha_sign = -1, corr_window = 10
            Explanation: long stocks with negative coorelation between volume and open price
            '''
            para1 = PARAMETER['alpha_6']['alpha_sign']
            para2 = PARAMETER['alpha_6']['corr_window']
            return para1 * correlation(df_open,df_volume,window = para2)

        def Alpha7():
            '''
            Formula: alpha_sign * ((adv20<volume)?((ts_rank(abs(delta(close,delta_step)),tsrank_window))*sign(delta(close,delta_step))):1)
            Parameter: alpha_sign = -1, delta_step = 7, tsrank_window = 60
            Explanation:
            '''
            para1 = PARAMETER['alpha_7']['alpha_sign']
            para2 = PARAMETER['alpha_7']['delta_step']
            para3 = PARAMETER['alpha_7']['tsrank_window']
            temp1 = df_nan.copy()
            temp2 = ts_rank(delta(df_close, days = para2).abs(),window = para3)
            temp3 = df_adv(20) < df_volume
            temp1[temp3] = temp2[temp3]
            temp4 = df_adv(20) > df_volume
            temp1[temp4] = 1
            return para1 * temp1

        def Alpha8():
            '''
            Formula: alpha_sign * rank(((sum(open,lookback_window)*sum(returns,lookback_window))-delay((sum(open,lookback_window)*sum(returns,lookback_window)),delay_step)))
            Parameter: alpha_sign = -1, lookback_window = 5, delay_step = 10
            Explanation: 
            '''
            para1 = PARAMETER['alpha_8']['alpha_sign']
            para2 = PARAMETER['alpha_8']['lookback_window']
            para3 = PARAMETER['alpha_8']['delay_step']
            temp1 = ts_sum(df_open,window = para2) * ts_sum(df_return, window = para2)
            return para1 * rank(temp1 - delay(temp1, para3))

        def Alpha9():
            '''
            Formula: ((0<ts_min(delta(close,delta_step),ts_window))?delta(close,delta_step)
                    :((ts_max(delta(close,delta_step),ts_window)<0)?delta(close,delta_step):(-1*delta(close,delta_step)))
            Parameter: ts_window = 5, delta_step = 1(should always be 1)
            Explanation: If price consistently moved up(down), we long(short) it wrt price change recently; 
                        otherwise, we hold position against the price change direction
            '''
            para1 = PARAMETER['alpha_9']['ts_window']
            para2 = PARAMETER['alpha_9']['delta_step']
            temp1 = delta(df_close, days = para2)
            temp3 = ts_min(temp1, window = para1)
            temp4 = ts_max(temp1, window = para1)
            cond1 = temp3 > 0
            cond2 = temp4 < 0
            temp2 = df_nan.copy()
            temp2[cond1] = temp1[cond1]
            temp2[cond2] = temp1[cond2]
            cond1 = temp3 <= 0
            cond2 = temp4 >= 0
            temp2[cond1] = -temp1[cond1]
            temp2[cond2] = -temp1[cond2]
            return temp2

        def Alpha10():
            '''
            Formula: rank(((0<ts_min(delta(close,delta_step),ts_window)?delta(close,delta_step):
                          ((ts_max(delta(close,delta_step),ts_window)<0)?delta(close,delta_step):(-1*delta(close,delta_step)))))
            Parameter: ts_window = 4, delta_step = 1(delta_step should always be 1)
            Explanation: Considering rank, if price consistently moved up(down), we long(short) it wrt price change recently; 
                        otherwise, we hold position against the price change direction
            '''
            para1 = PARAMETER['alpha_10']['ts_window']
            para2 = PARAMETER['alpha_10']['delta_step']
            temp1 = delta(df_close, days = para2)
            temp3 = ts_min(temp1, window = para1)
            temp4 = ts_max(temp1, window = para1)
            cond1 = temp3 > 0
            cond2 = temp4 < 0
            temp2 = df_nan.copy()
            temp2[cond1] = temp1[cond1]
            temp2[cond2] = temp1[cond2]
            cond1 = temp3 <= 0
            cond2 = temp4 >= 0
            temp2[cond1] = -temp1[cond1]
            temp2[cond2] = -temp1[cond2]
            return rank(temp2)
        
        def Alpha11():
            '''
            Formula: ((rank(ts_max((vwap-close),lookback_window))+rank(ts_min((vwap-close),lookback_windwo)))*rank(delta(volume,lookback_window)))
            Parameter: lookback_window = 3
            Explanation:
            '''
            para1 = PARAMETER['alpha_11']['lookback_window']
            temp = df_vwap - df_close
            return (rank(ts_max(temp,para1))+ rank(ts_min(temp,para1))) * rank(delta(df_volume,para1))

        def Alpha12():
            '''
            Formula: (sign(delta(volume,delta_step))*(alpha_sign*delta(close,delta_step)))
            Parameter: delta_step = 1, alpha_sign = -1
            Explanation: long stocks that has decreasing volume/price, increasing price/volume, short otherwise
            '''
            para1 = PARAMETER['alpha_12']['delta_step']
            para2 = PARAMETER['alpha_12']['alpha_sign']
            return np.sign(delta(df_volume,para1)) * para2 * delta(df_close,para1)

        def ts_cov(df1,df2,window = 10):
            return df1.rolling(window = window).cov(df2)
        def Alpha13():
            '''
            Formula: alpha_sign * rank(covariance(rank(close),rank(volume),cov_window))
            Parameter: alpha_sign = -1, cov_window = 5
            Explanation: long more stable stocks wrt close and volume covariance 
            '''
            para1 = PARAMETER['alpha_13']['alpha_sign']
            para2 = PARAMETER['alpha_13']['cov_window']
            return para1*ts_cov(rank(df_close),rank(df_volume),para2)

        def Alpha14():
            '''
            Formula: alpha_sign * rank(delta(returns,delta_step))*correlation(open,volume,corr_window)
            Parameter: alpha_sign = -1, delta_step = 3, corr_window = 5
            Explanation: 
            '''
            para1 = PARAMETER['alpha_14']['alpha_sign']
            para2 = PARAMETER['alpha_14']['delta_step']
            para3 = PARAMETER['alpha_14']['corr_window']
            return para1 * rank(delta(df_return,para2)) * correlation(df_open,df_volume,para3)

        def Alpha15():
            '''
            Formula: alpha_sign * sum(rank(correlation(rank(high),rank(volume),corr_window)),sum_window)
            Parameter: alpha_sign = -1, corr_window = 3, sum_window = 3
            Explanation:
            '''
            para1 = PARAMETER['alpha_15']['alpha_sign']
            para2 = PARAMETER['alpha_15']['corr_window']
            para3 = PARAMETER['alpha_15']['sum_window']
            return para1 * ts_sum(rank(correlation(rank(df_high),rank(df_volume),para2)),para3)

        def Alpha16():
            '''
            Formula: alpha_sign * rank(covariance(rank(high),rank(volume),cov_window))
            Parameter: alpha_sign = -1, cov_window = 5
            Explanation: Similar to Alpha13
            '''
            para1 = PARAMETER['alpha_16']['alpha_sign']
            para2 = PARAMETER['alpha_16']['cov_window']
            return para1 * rank(ts_cov(rank(df_high),rank(df_volume),para2))

        def Alpha17():
            '''
            Formula: alpha_sign * rank(ts_rank(close,tsrank_close_window)) * rank(delta(delta(close,delta_step),delta_step)) * rank(ts_rank((volume/adv20),tsrank_volume_window))
            Parameter: alpha_sign = -1, delta_step = 1, tsrank_close_window = 10, tsrank_volume_window = 5
            Explanation:
            '''
            para1 = PARAMETER['alpha_17']['alpha_sign']
            para2 = PARAMETER['alpha_17']['delta_step']
            para3 = PARAMETER['alpha_17']['tsrank_close_window']
            para4 = PARAMETER['alpha_17']['tsrank_volume_window']
            return para1 * rank(ts_rank(df_close,para3)) * rank(delta(delta(df_close,para2),para2)) * rank(ts_rank(df_volume/df_adv(20),para4))

        def Alpha18():
            '''
            Formula: alpha_sign * rank(((stddev(abs(close-open),std_window)+(close-open))+correlation(close,open,corr_window)))
            Parameter: alpha_sign = -1, std_window = 5, corr_window = 10
            Explanation:
            '''
            para1 = PARAMETER['alpha_18']['alpha_sign']
            para2 = PARAMETER['alpha_18']['std_window']
            para3 = PARAMETER['alpha_18']['corr_window']
            temp = df_close - df_open
            return rank(ts_stddev(np.abs(temp),para2)+ temp + correlation(df_close,df_open,para3)) * para1

        def Alpha19():
            '''
            Formula: alpha_sign * sign(delta(close,delta_step)) * (1+ rank(sum(returns,sum_window)))
            Parameter: alpha_sign = -1, delta_step = 7, sum_window = 250
            Explanation: long stocks that price lower than last week, share depend on average return rank
            '''
            para1 = PARAMETER['alpha_19']['alpha_sign']
            para2 = PARAMETER['alpha_19']['delta_step']
            para3 = PARAMETER['alpha_19']['sum_window']
            return para1 * np.sign(delta(df_close,para2)) * (1 + rank(ts_sum(df_return,para3)))

        def Alpha20():
            '''
            Formula: alpha_sign * rank(open-delay(high,lookback_window))*rank(open-delay(close,lookback_window))*rank(open-delay(low,lookback_window))
            Parameter: alpha_sign = -1, lookback_window = 1
            Explanation: consider today open price with yesterday high, close, low price
            '''
            para1 = PARAMETER['alpha_20']['alpha_sign']
            para2 = PARAMETER['alpha_20']['lookback_window']
            return para1 * rank(df_open - delay(df_high,para2)) * rank(df_open - delay(df_close,para2)) * rank(df_open - delay(df_low,para2))              
        
        def Alpha21():
            '''
            Formula: (mean(close,lookback_window)+stddev(close,lookback_window)) < mean(close,current_window) ? alpha_sign :
                     (mean(close,current_window)) < (mean(close,lookback_window)-stddev(close,lookback_window))? -alpha_sign:
                     (adv20<=volume)? -alpha_sign : alpha_sign
            Parameter: alpha_sign = -1, lookback_window = 8, current_window = 2
            Explanation: short if longterm prediction upperbound is lower than shorterm mean,
                         long if longterm prediction lowerbound is higher than shorterm mean,
                         otherwise long if volume no less than adv20, short otherwise 
            '''
            para1 = PARAMETER['alpha_21']['alpha_sign']
            para2 = PARAMETER['alpha_21']['lookback_window']
            para3 = PARAMETER['alpha_21']['current_window']
            temp1 = ts_mean(df_close,para2) + ts_stddev(df_close,para2)
            temp2 = ts_mean(df_close,para2) - ts_stddev(df_close,para2)
            bench = ts_mean(df_close,para3)
            ans = df_nan.copy()
            cond1 = temp1 < bench
            cond2 = temp2 > bench
            cond3 = df_adv(20) <= df_volume
            cond4 = df_adv(20) > df_volume
            ans[cond4] = para1
            ans[cond3] = -para1
            ans[cond2] = -para1
            ans[cond1] = para1
            return ans

        def Alpha22():
            '''
            Formula: alpha_sign * delta(correlation(high,volume,corr_window),delta_step) * rank(stddev(close,std_window))
            Parameter: alpha_sign = -1, corr_window = 5, delta_step = 5, std_window = 20
            Explanation: 
            '''
            para1 = PARAMETER['alpha_22']['alpha_sign']
            para2 = PARAMETER['alpha_22']['corr_window']
            para3 = PARAMETER['alpha_22']['delta_step']
            para4 = PARAMETER['alpha_22']['std_window']
            return para1 * delta(correlation(df_high,df_volume,para2),para3) * rank(ts_stddev(df_close,para4))

        def Alpha23():
            '''
            Formula: (mean(high,mean_window)<high)? alpha_sign * delta(high,delta_step) : 0
            Parameter: alpha_sign = -1, mean_window = 20, delta_step = 2
            Explanation: short if high price higher than yesterday and higher than mean, 
                         long if high price lower than yesterday and higher than mean.
            '''
            para1 = PARAMETER['alpha_23']['alpha_sign']
            para2 = PARAMETER['alpha_23']['mean_window']
            para3 = PARAMETER['alpha_23']['delta_step']
            temp1 = ts_mean(df_high,para2)
            cond1 = temp1 <  df_high
            cond2 = temp1 >= df_high
            ans = df_nan.copy()
            ans[cond1] = para1 * delta(df_high,para3)
            ans[cond2] = 0
            return ans

        def Alpha24():
            '''
            Formula: (((delta(mean(close,mean_window),change_window))/delay(close,change_window))<= pct_change) ? 
                        alpha_sign * (close - ts_min(close, min_window)) : alpha_sign * delta(close,delta_step)
            Parameter: alpha_sign = -1, change_window = 100, mean_window = 100, pct_change = 0.05, delta_step = 3, min_window = 100
            Explanation: 
            '''
            para1 = PARAMETER['alpha_24']['alpha_sign']
            para2 = PARAMETER['alpha_24']['mean_window']
            para3 = PARAMETER['alpha_24']['pct_change']
            para4 = PARAMETER['alpha_24']['delta_step']
            para5 = PARAMETER['alpha_24']['change_window']
            para6 = PARAMETER['alpha_24']['min_window']
            temp1 = delta(ts_mean(df_close,para2),para5) / delay(df_close,para5)
            cond1 = temp1<= para3
            cond2 = temp1 > para3
            ans = df_nan.copy()
            ans[cond1] = para1 * (df_close - ts_min(df_close,para6))
            ans[cond2] = para1 * delta(df_close,para4)
            return ans

        def Alpha25():
            '''
            Formula: rank(alpha_sign * returns * adv20 * vwap * (high - close))
            Parameter: alpha_sign = -1
            Explanation: favor stock with less volatility
            '''
            para1 = PARAMETER['alpha_25']['alpha_sign']
            return rank(para1 * df_return * df_adv(20) * df_vwap * (df_high - df_close))

        def Alpha26():
            '''
            Formula: alpha_sign * ts_max(correlation(ts_rank(volume,tsrank_window),ts_rank(high,tsrank_window),corr_window),tsmax_window)
            Parameter: alpha_sign = -1, tsrank_window = 5, corr_window = 5, tsmax_window = 3
            Explanation:
            '''
            para1 = PARAMETER['alpha_26']['alpha_sign']
            para2 = PARAMETER['alpha_26']['tsrank_window']
            para3 = PARAMETER['alpha_26']['corr_window']
            para4 = PARAMETER['alpha_26']['tsmax_window']
            return para1 * ts_max(correlation(ts_rank(df_volume,para2),ts_rank(df_high,para2),para3),para4)

        def Alpha27():
            '''
            Formula: (rank_pct <rank((mean(correlation(rank(volume),rank(vwap),corr_window),mean_window))))? alpha_sign:-alpha_sign
            Parameter: rank_pct = 0.5, corr_window = 6, mean_window = 2, alpha_sign = -1
            Explanation:
            '''
            para1 = PARAMETER['alpha_27']['alpha_sign']
            para2 = PARAMETER['alpha_27']['corr_window']
            para3 = PARAMETER['alpha_27']['rank_pct']
            para4 = PARAMETER['alpha_27']['mean_window']
            temp = rank(ts_mean(correlation(rank(df_volume),rank(df_vwap),para2),para4))
            cond1 = temp > para3
            cond2 = temp <= para3
            ans = df_nan.copy()
            ans[cond1] = para1
            ans[cond2] = -para1
            return ans

        def Alpha28():
            '''
            Formula: scale(correlation(adv20,low,corr_window)+(high + low) * 0.5 - close)
            Parameter: corr_window = 5
            Explanation:
            '''
            para1 = PARAMETER['alpha_28']['corr_window']
            return scale(correlation(df_adv(20),df_low,para1)+ (df_high + df_low) * 0.5 - df_close)

        def Alpha29():
            '''
            Formula:
        (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((alpha_sign * rank(delta(close,5))))),2),1))))),1),5)+ts_rank(delay(alpha_sign * returns,6),5))
            Parameter: delta_step = 5, tsmin_window = 2, sum_window = 1, prod_window = 1, min_window = 5, delay_step = 6, tsrank_window = 5, alpha_sign = -1
            Explanation:
            '''
            para1 = PARAMETER['alpha_29']['delta_step']
            para2 = PARAMETER['alpha_29']['tsmin_window']
            para3 = PARAMETER['alpha_29']['sum_window']
            para4 = PARAMETER['alpha_29']['prod_window']
            para5 = PARAMETER['alpha_29']['min_window']
            para6 = PARAMETER['alpha_29']['delay_step']
            para7 = PARAMETER['alpha_29']['tsrank_window']
            para8 = PARAMETER['alpha_29']['alpha_sign']
            temp1 = scale(Log(ts_sum(ts_min(rank(rank(para8 * rank(delta(df_close, para1)))),para2),para3)))
            temp2 = ts_min(ts_product(rank(rank(temp1)),para4),para5)
            temp3 = ts_rank(delay(para8 * df_return,para6),para7)
            return temp2 + temp3

        def Alpha30():
            '''
            Formula: (1.0 - rank(sign(delta(close,1))+sign(delay(delta(close,1),1))+...+sign(delay(delta(close,1),delat_steps)))) * sum(volume,5)/sum(volume,20)
            Parameter: delta_step = 1, delay_steps = 2, sum_window1 = 5, sum_window2 = 20
            Explanation:
            '''
            para1 = PARAMETER['alpha_30']['delta_step']
            para2 = PARAMETER['alpha_30']['delay_steps']
            para3 = PARAMETER['alpha_30']['sum_window1']
            para4 = PARAMETER['alpha_30']['sum_window2']
            temp = np.sign(delta(df_close,para1))
            temp_sum = rank(ts_sum(temp,para2+1))
            return (1-temp_sum) * ts_sum(df_volume,para3) / ts_sum(df_volume,para4)

        def Alpha31():
            '''
            Formula: rank(  rank(rank(decay_linear((alpha_sign * rank(rank(delta(close, delta_step)))), decay_window))) 
                            + rank((alpha_sign * delta(close, delta_step2)))  
                            + sign(scale(correlation(adv20, low, corr_window))))
            Parameter: alpha_sign = -1, delta_step = 10 , decay_window = 10, delta_step2 = 3, corr_window = 20
            Explanation:
            '''
            para1 = PARAMETER['alpha_31']['alpha_sign']
            para2 = PARAMETER['alpha_31']['delta_step']
            para3 = PARAMETER['alpha_31']['decay_window']
            para4 = PARAMETER['alpha_31']['delta_step2']
            para5 = PARAMETER['alpha_31']['corr_window']
            temp1 = rank(rank(delta(df_close,para2))) * para1
            temp1 = rank(decay_linear(temp1,para3))
            temp2 = rank(delta(df_close,para4) * para1)
            temp3 = np.sign(scale(correlation(df_adv(20),df_low,para5)))
            return temp1+temp2+temp3

        def Alpha32():
            '''
            Formula: scale((mean(close,mean_window)-close)) 
                     + scale_para * scale(correlation(vwap,delay(close,delay_step),corr_window))
            Parameter: mean_window = 7, scale_para = 20, delay_step = 5, corr_window = 230
            Explanation:
            '''
            para1 = PARAMETER['alpha_32']['mean_window']
            para2 = PARAMETER['alpha_32']['scale_para']
            para3 = PARAMETER['alpha_32']['delay_step']
            para4 = PARAMETER['alpha_32']['corr_window']
            temp1 = scale(ts_mean(df_close,para1)-df_close)
            temp2 = para2 * scale(correlation(df_vwap,delay(df_close,para3),para4))
            return temp1 + temp2

        def Alpha33():
            '''
            Formula: rank(alpha_sign * (1-(open/close))^pow_level)
            Parameter: alpha_sign = -1, pow_level = 1
            Explanation: Favor lower daily price change
            '''
            para1 = PARAMETER['alpha_33']['alpha_sign']
            para2 = PARAMETER['alpha_33']['pow_level']
            return rank(para1 * SignedPower(1 - df_open / df_close , para2))

        def Alpha34():
            '''
            Formula: rank((1 - rank((stddev(returns, std_window1) / stddev(returns, std_window2))) + 1 - rank(delta(close, delta_step))))
            Parameter: std_window1 = 2, std_window2 = 5, delta_step = 1
            Explanation: 
            '''
            para1 = PARAMETER['alpha_34']['std_window1']
            para2 = PARAMETER['alpha_34']['std_window2']
            para3 = PARAMETER['alpha_34']['delta_step']
            temp1 = 1 - rank(ts_stddev(df_return,para1)/ts_stddev(df_return,para2))
            temp2 = 1 - rank(delta(df_close,para3))
            return rank(temp1 + temp2)

        def Alpha35():
            '''
            Formula: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
            Parameter: tsrank_window1 = 32, tsrank_window2 = 16, tsrank_window3 = 32
            Explanation:
            '''
            para1 = PARAMETER['alpha_35']['tsrank_window1']
            para2 = PARAMETER['alpha_35']['tsrank_window2']
            para3 = PARAMETER['alpha_35']['tsrank_window3']
            temp1 = ts_rank(df_volume,para1)
            temp2 = 1 - ts_rank(df_close + df_high - df_low, para2)
            temp3 = 1 - ts_rank(df_return,para3)
            return temp1 * temp2 * temp3

        def Alpha36():
            '''
            Formula: (  coef1 * rank(correlation((close - open), delay(volume, delay_step), corr_window)) 
                        + coef2 * rank((open - close)) 
                        + coef3 * rank(Ts_Rank(delay((alpha_sign * returns), delay_step2), tsrank_window)) 
                        + coef4 * rank(abs(correlation(vwap, adv20, corr_window2))) 
                        + coef5 * rank((mean(close, mean_window) - open) * (close - open)) )
            Parameter: coef1 = 2.21, delay_step = 1, corr_window = 15,
                       coef2 = 0.7, 
                       coef3 = 0.73, alpha_sign = -1, delay_step2 = 6, tsrank_window = 5,
                       coef4 = 1, corr_window2 = 6
                       coef5 = 0.6, mean_window = 200
            Explanation:
            '''
            para1_1 = PARAMETER['alpha_36']['coef1']
            para1_2 = PARAMETER['alpha_36']['delay_step']
            para1_3 = PARAMETER['alpha_36']['corr_window']
            para2 = PARAMETER['alpha_36']['coef2']
            para3_1 = PARAMETER['alpha_36']['coef3']
            para3_2 = PARAMETER['alpha_36']['alpha_sign']
            para3_3 = PARAMETER['alpha_36']['delay_step2']
            para3_4 = PARAMETER['alpha_36']['tsrank_window']
            para4_1 = PARAMETER['alpha_36']['coef4']
            para4_2 = PARAMETER['alpha_36']['corr_window2']
            para5_1 = PARAMETER['alpha_36']['coef5']
            para5_2 = PARAMETER['alpha_36']['mean_window']

            temp1 = para1_1 * rank(correlation(df_close-df_open,delay(df_volume,para1_2),para1_3))
            temp2 = para2 * rank(df_open - df_close)
            temp3 = para3_1 * rank(ts_rank(delay(para3_2 * df_return, para3_3), para3_4))
            temp4 = para4_1 * rank(np.abs(correlation(df_vwap,df_adv(20),para4_2)))
            temp5 = para5_1 * rank((ts_mean(df_close,para5_2) - df_open) * (df_close - df_open))
            return temp1 + temp2 + temp3 + temp4 + temp5

        def Alpha37():
            '''
            Formula: rank(correlation(delay((open - close), delay_step), close, corr_window)) 
                     + rank(open - close)
            Parameter: delay_step = 1, corr_window = 200
            Explanation:
            '''
            para1 = PARAMETER['alpha_37']['delay_step']
            para2 = PARAMETER['alpha_37']['corr_window']
            temp1 = df_open - df_close
            part1 = rank(correlation(delay(temp1,para1),df_close,para2))
            part2 = rank(temp1)
            return part1 + part2

        def Alpha38():
            '''
            Formula: alpha_sign * rank(Ts_Rank(close, 10)) * rank(close / open)
            Parameter: alpha_sign = -1, tsrank_window = 10
            Explanation:
            '''
            para1 = PARAMETER['alpha_38']['alpha_sign']
            para2 = PARAMETER['alpha_38']['tsrank_window']
            return para1 * rank(ts_rank(df_close, para2)) * rank(df_close / df_open)

        def Alpha39():
            '''
            Formula: (alpha_sign * rank( delta(close, delta_step) * (1 - rank(decay_linear(volume / adv20, decay_window))) )
                      * (1 + rank(sum(returns, sum_window))))
            Parameter: alpha_sign = -1, delta_step = 7, decay_window = 9, sum_window = 250    
            Explanation:
            '''
            para1 = PARAMETER['alpha_39']['alpha_sign']
            para2 = PARAMETER['alpha_39']['delta_step']
            para3 = PARAMETER['alpha_39']['decay_window']
            para4 = PARAMETER['alpha_39']['sum_window']
            temp1 = delta(df_close, para2) * (1 - rank(decay_linear(df_volume / df_adv(20), para3)))
            temp1 = rank(temp1)
            temp2 = 1 + rank(ts_sum(df_return,para4))
            return para1 * temp1 * temp2

        def Alpha40():
            '''
            Formula: (alpha_sign * rank(stddev(high, lookback_window))) * correlation(high, volume, lookback_window)
            Parameter: alpha_sign = -1, lookback_window = 10
            Explanation:
            '''
            para1 = PARAMETER['alpha_40']['alpha_sign']
            para2 = PARAMETER['alpha_40']['lookback_window']
            temp1 = rank(ts_stddev(df_high,para2))
            temp2 = correlation(df_high, df_volume, para2)
            return para1 * temp1 * temp2        
        
        def Alpha41():
            '''
            Formula: (((high * low)^0.5) - vwap)
            Explanation: 
            '''
            return np.sqrt(df_high*df_low) - df_vwap

        def Alpha42():
            '''
            Formula: (rank((vwap - close)) / rank((vwap + close)))
            Explanation: 
            '''
            res=rank(df_vwap-df_close) / rank(df_vwap+df_close)
            return res
       
        def Alpha43():
            '''
            Formula: (ts_rank((volume / adv(vol_window)), vol_window) * ts_rank((-1 * delta(close, close_window-1)), close_window))
            Parameter: {'vol_window':20,'close_window':8}
            Explanation: 
            '''
            vol_window = PARAMETER['alpha_43']['vol_window']
            close_window = PARAMETER['alpha_43']['close_window']
            res=ts_rank(df_volume/df_adv(vol_window), vol_window) * ts_rank(-1*delta(df_close, close_window-1), close_window)
            return res
        def Alpha44():
            '''
            Formula: (-1 * correlation(high, rank(volume), corr_window))
            Parameter: {'corr_window':5}
            Explanation:
            '''
            corr_window = PARAMETER['alpha_44']['corr_window']
            res=(-1) * correlation(df_high, rank(df_volume), corr_window)
            return res

        def Alpha45():
            '''
            Formula: (-1 * ((rank((sum(delay(close, delay_window), sum_window) / sum_window)) 
                         * correlation(close, volume, corr_window)) 
                         * rank(correlation(sum(close, delay_window), sum(close, sum_window), corr_window))))
            Parameter: {'delay_window':5,'sum_window':20,'corr_window':2}
            Explanation:
            '''
            delay_window = PARAMETER['alpha_45']['delay_window']
            sum_window = PARAMETER['alpha_45']['sum_window']
            corr_window = PARAMETER['alpha_45']['corr_window']
            p1 = rank(ts_mean(delay(df_close, delay_window), sum_window))
            p2 = correlation(df_close, df_volume, corr_window)
            p3 = rank(correlation(ts_sum(df_close, delay_window), ts_sum(df_close, sum_window), corr_window))
            return (-1) * p1 * p2 * p3

        def Alpha46():
            '''
            Formula: ((0.25 < (((delay(close, lookback_window*2) - delay(close, lookback_window)) / lookback_window) - ((delay(close, lookback_window) - close) / lookback_window))) ? (-1 * 1) : (((((delay(close, lookback_window*2) - delay(close, lookback_window)) / lookback_window) - ((delay(close, lookback_window) - close) / lookback_window)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
            Parameter: {'lookback_window':10}
            Explanation:
            '''
            lookback_window = PARAMETER['alpha_46']['lookback_window']
            temp1 = (delay(df_close, 2*lookback_window) - delay(df_close, lookback_window)) / lookback_window
            temp2 = (delay(df_close, lookback_window) - df_close) / lookback_window
            mask1 = temp1 - temp2 < 0
            mask2 = temp1 - temp2 > 0.25
            mask3 = ((temp1 - temp2) >= 0) & ((temp1 - temp2) <= 0.25)
            resDf = df_nan.copy()
            resDf[mask1] = 1
            resDf[mask2] = -1
            resDf[mask3] = (-1) * (df_close - delay(df_close, 1))
            return resDf

        def Alpha47():
            '''
            Formula: ((((rank((1 / close)) * volume) / adv(vol_window)) * ((high * rank((high - close))) / (sum(high, sum_window) / sum_window))) - rank((vwap - delay(vwap, delay_window))))
            Parameter: {'vol_window':20,'sum_window':5,'delay_window':5}
            Explanation:
            '''
            vol_window = PARAMETER['alpha_47']['vol_window']
            sum_window = PARAMETER['alpha_47']['sum_window']
            delay_window = PARAMETER['alpha_47']['delay_window']
            p1 = rank(df_close**(-1)) * df_volume / df_adv(vol_window)
            p2 = df_high * rank(df_high - df_close) / ts_mean(df_high, sum_window)
            p3 = ((df_vwap - delay(df_vwap, delay_window)))
            return p1 * p2 - p3
        
        def Alpha48():
            '''
            Formula: (indneutralize(((correlation(delta(close, short_window), delta(delay(close, short_window), short_window), long_window) 
                    * delta(close, short_window)) / close), IndClass.subindustry) / sum(((delta(close, short_window) / delay(close, short_window))^2), long_window))
            Parameter: {'short_window':1,'long_window':250}
            Explanation:
            '''
            short_window = PARAMETER['alpha_48']['short_window']
            long_window = PARAMETER['alpha_48']['long_window']
            temp = correlation(delta(df_close, short_window), delta(delay(df_close, short_window), short_window), long_window)*delta(df_close, short_window) / df_close
            p1 = IndNeutralize(temp, df_ind['Subindustry'])
            p2 = ts_sum((delta(df_close, short_window) / delay(df_close, short_window))**2, long_window)
            return p1 / p2
            
        def Alpha49():
            '''
            Formula: (((((delay(close, 2*lookback_window) - delay(close, lookback_window)) / lookback_window) - ((delay(close, lookback_window) - close) / lookback_window)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
            Parameter: {'lookback_window':10}
            Explanation:
            '''
            lookback_window = PARAMETER['alpha_49']['lookback_window']
            temp1 = (delay(df_close, 2*lookback_window) - delay(df_close, lookback_window)) / lookback_window
            temp2 = (delay(df_close, lookback_window) - df_close) / lookback_window
            mask = temp1 - temp2 < -0.1
            resDf = (-1) * (df_close - delay(df_close, 1))
            resDf[mask] = 1
            return resDf
            
        def Alpha50():
            '''
            Formula: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), lookback_window)), lookback_window))
            Parameter: {'lookback_window':5}
            Explanation:
            '''
            lookback_window = PARAMETER['alpha_50']['lookback_window']
            res=(-1) * ts_max(rank(correlation(rank(df_volume), rank(df_vwap), lookback_window)), lookback_window)
            return res

        def Alpha51():
            '''
            Formula: (((((delay(close, 2*lookback_window) - delay(close, lookback_window)) / lookback_window) - ((delay(close, lookback_window) - close) / lookback_window)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
            Parameter: {'lookback_window':10}
            Explanation:
            '''
            lookback_window = PARAMETER['alpha_51']['lookback_window']
            temp1 = (delay(df_close, 2*lookback_window) - delay(df_close, lookback_window)) / lookback_window
            temp2 = (delay(df_close, lookback_window) - df_close) / lookback_window
            mask1 = temp1 - temp2 < -0.05
            mask2 = temp1 - temp2 >= -0.05
            resDf = df_nan.copy()
            resDf[mask1] = 1
            resDf[mask2] = (-1) * (df_close - delay(df_close, 1))
            return resDf
            
        def Alpha52():
            '''
            Formula: ((((-1 * ts_min(low, lookback_window)) + delay(ts_min(low, lookback_window), lookback_window)) 
                        * rank(((sum(returns, long_sum_window) - sum(returns, short_sum_window)) / (long_sum_window-short_sum_window)))) * ts_rank(volume, lookback_window))
            Parameter: {'lookback_window':5,'long_sum_window':240,'short_sum_window':20}
            Explanation:
            '''
            lookback_window = PARAMETER['alpha_52']['lookback_window']
            long_sum_window = PARAMETER['alpha_52']['long_sum_window']
            short_sum_window = PARAMETER['alpha_52']['short_sum_window']
            p1 = (-1) * ts_min(df_low, lookback_window) + delay(ts_min(df_low, lookback_window), lookback_window)
            p2 = rank((ts_sum(df_return, short_sum_window)-ts_sum(df_return, short_sum_window))/(long_sum_window-short_sum_window))
            p3 = ts_rank(df_volume, lookback_window)
            return p1 * p2 * p3

        def Alpha53():
            '''
            Formula: (-1 * delta((((close - low) - (high - close)) / (close - low)), lookback_window))
            Parameter: {'lookback_window':9}
            Explanation:
            '''
            lookback_window = PARAMETER['alpha_53']['lookback_window']
            res=(-1) * delta(((df_close-df_low)-(df_high-df_close))/(df_close-df_low), lookback_window) 
            return res

        def Alpha54():
            '''
            Formula: ((-1 * ((low - close) * (open^price_power))) / ((low - high) * (close^price_power)))
            Parameter: {'price_power':5}
            Explanation:
            '''
            price_power = PARAMETER['alpha_54']['price_power']
            p1 = (-1) * (df_low-df_close) * df_open**price_power
            p2 = (df_low-df_high) * df_close**price_power
            return p1 / p2
            
        def Alpha55():
            '''
            Formula: (-1 * correlation(rank(((close - ts_min(low, minmax_window)) / 
                     (ts_max(high, minmax_window) - ts_min(low, minmax_window)))), rank(volume), corr_window))
            Parameter: {'minmax_window':12,'corr_window':6}
            Explanation:
            '''
            minmax_window = PARAMETER['alpha_55']['minmax_window']
            corr_window = PARAMETER['alpha_55']['corr_window']
            temp1 = df_close - ts_min(df_low, minmax_window)
            temp2 = ts_max(df_high, minmax_window) - ts_min(df_low, minmax_window)
            res=(-1) * correlation(rank(temp1/temp2), rank(df_volume), corr_window)
            return res

        def Alpha56():
            '''
            Formula:  -1 * (rank((sum(returns, long_window) / sum(sum(returns, short_window), mid_window))) * rank((returns * cap)))
            Parameter: {'long_window':10,'mid_window':3,'short_window':2,'cap_window':20}
            Explanation:
            '''
            long_window = PARAMETER['alpha_56']['long_window']
            mid_window = PARAMETER['alpha_56']['mid_window']
            short_window = PARAMETER['alpha_56']['short_window']
            cap_window = PARAMETER['alpha_56']['cap_window']
            res=(-1) * (rank((ts_sum(df_return, long_window) / ts_sum(ts_sum(df_return, short_window), mid_window))) * rank((df_return * df_cap(cap_window))))
            return res

        def Alpha57():
            '''
            Formula: (close - vwap) / decay_linear(rank(ts_argmax(close, lookback_window)), decay_window)
            Parameter: {'lookback_window':30,'decay_window':2}
            Explanation:
            '''
            lookback_window = PARAMETER['alpha_57']['lookback_window']
            decay_window = PARAMETER['alpha_57']['decay_window']
            res=(df_close - df_vwap) / decay_linear(rank(ts_argmax(df_close, lookback_window)), decay_window)
            return res

        def Alpha58():
            '''
            Formula: -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, corr_window), decay_window), rank_window)
            Parameter: {'corr_window':3,'decay_window':7,'rank_window':5}
            Explanation:
            '''
            corr_window = PARAMETER['alpha_58']['corr_window']
            decay_window = PARAMETER['alpha_58']['decay_window']
            rank_window = PARAMETER['alpha_58']['rank_window']
            temp = correlation(IndNeutralize(df_vwap, df_ind['Sector']), df_volume, corr_window)
            res=(-1)*ts_rank(decay_linear(temp, decay_window), rank_window)
            return res

        def Alpha59():
            '''
            Formula: -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * factor) + (vwap * (1 - factor))), IndClass.industry), volume, corr_window), decay_window), rank_window)
            Parameter: {'factor':0.728317,'corr_window':4,'decay_window':16,'rank_window':8}
            Explanation:
            '''
            factor = PARAMETER['alpha_59']['factor']
            corr_window = PARAMETER['alpha_59']['corr_window']
            decay_window = PARAMETER['alpha_59']['decay_window']
            rank_window = PARAMETER['alpha_59']['rank_window']
            temp1 = df_vwap * factor + df_vwap * (1 - factor)
            temp2 = correlation(IndNeutralize(temp1, df_ind['Industry']), df_volume, corr_window)
            res=(-1)*ts_rank(decay_linear(temp2, decay_window), rank_window)
            return res

        def Alpha60():
            '''
            Formula: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) 
                        - scale(rank(ts_argmax(close, lookback_window))))))
            Parameter: {'lookback_window':10}
            Explanation:
            '''
            lookback_window = PARAMETER['alpha_60']['lookback_window']
            temp1 = rank(((df_close-df_low)-(df_high-df_close))/(df_high-df_low)*df_volume)
            temp2 = rank(ts_argmax(df_close, lookback_window))
            res=(-1) * (2*scale(temp1) - scale(temp2))
            return res

        def Alpha61():
            '''
            Formula: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282))))
            '''
            minwindow = PARAMETER['alpha_61']['minwindow']
            corrwindow = PARAMETER['alpha_61']['corrwindow']
            return Df_Int(rank((df_vwap - ts_min(df_vwap, minwindow))) < rank(correlation(df_vwap, df_adv(180), corrwindow)))
            
        def Alpha62(): 
            '''
            Formula: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +
rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
            ''' 
            sumwindow = PARAMETER['alpha_62']['sumwindow']
            corrwindow = PARAMETER['alpha_62']['corrwindow']
            return Df_Int((rank(correlation(df_vwap, ts_sum(df_adv(20), sumwindow), corrwindow)) < Df_Int(rank(((rank(df_open) + rank(df_open)) < (rank(((df_high + df_low) / 2)) + rank(df_high))))))) * (-1)
            
        def Alpha63():
            '''
            Formula: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))
- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,
37.2467), 13.557), 12.2883))) * -1)
            '''
            deltawindow = PARAMETER['alpha_63']['deltawindow']
            decaywindow = PARAMETER['alpha_63']['decaywindow']
            weight = PARAMETER['alpha_63']['weight']
            sumwindow = PARAMETER['alpha_63']['sumwindow']
            corrwindow = PARAMETER['alpha_63']['corrwindow']
            decaywindow2 = PARAMETER['alpha_63']['decaywindow2']
            return (rank(decay_linear(delta(IndNeutralize(df_close, df_ind['Industry']), deltawindow), decaywindow))- rank(decay_linear(correlation(((df_vwap * weight) + (df_open * (1 - weight))), ts_sum(df_adv(180),sumwindow), corrwindow), decaywindow2))) * (-1)
            

        def Alpha64():
            '''
            Formula:((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),
sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -
0.178404))), 3.69741))) * -1)
            '''
            weight = PARAMETER['alpha_64']['weight']
            sumwindow = PARAMETER['alpha_64']['sumwindow']
            corrwindow = PARAMETER['alpha_64']['corrwindow']
            weight2 = PARAMETER['alpha_64']['weight2']
            deltawindow = PARAMETER['alpha_64']['deltawindow']
            return Df_Int(rank(correlation(ts_sum(((df_open * weight) + (df_low * (1 - weight))), sumwindow),ts_sum(df_adv(120), sumwindow), corrwindow)) < rank(delta(((((df_high + df_low) / 2) * weight2) + (df_vwap * (1 -weight2))), deltawindow))) *( -1)
        
        def Alpha65():
            '''
            Formula:((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,
8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
            '''
            weight = PARAMETER['alpha_65']['weight']
            sumwindow = PARAMETER['alpha_65']['sumwindow']
            corrwindow = PARAMETER['alpha_65']['corrwindow']
            minwindow = PARAMETER['alpha_65']['minwindow']
            return Df_Int(rank(correlation(((df_open * weight) + (df_vwap * (1 - weight))), ts_sum(df_adv(60),sumwindow), corrwindow)) < rank((df_open - ts_min(df_open, minwindow)))) * (-1)
        
        def Alpha66():
            '''
            Formula:((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low
* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
            '''
            decaywindow1 = PARAMETER['alpha_66']['decaywindow1']
            weight = PARAMETER['alpha_66']['weight']
            deltawindow = PARAMETER['alpha_66']['deltawindow']
            decaywindow2 = PARAMETER['alpha_66']['decaywindow2']
            rankwindow = PARAMETER['alpha_66']['rankwindow']
            return (rank(decay_linear(delta(df_vwap, deltawindow),decaywindow1)) + ts_rank(decay_linear(((((df_low* weight) + (df_low * (1 - weight))) - df_vwap) / (df_open - ((df_high + df_low) / 2))), decaywindow2), rankwindow)) * (-1)
            
        def Alpha67():
            '''
            Formula:((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,
IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
            '''
            rankwindow = PARAMETER['alpha_67']['rankwindow']
            corrwindow = PARAMETER['alpha_67']['corrwindow']
            return (rank((df_high - ts_min(df_high, rankwindow)))**rank(correlation(IndNeutralize(df_vwap,df_ind['Sector']), IndNeutralize(df_adv(20), df_ind['Subindustry']), corrwindow))) * (-1)
            
        def Alpha68(): #dododododododo
            '''
            Formula:((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <
rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
            '''
            corrwindow = PARAMETER['alpha_68']['corrwindow']
            rankwindow = PARAMETER['alpha_68']['rankwindow']
            weight = PARAMETER['alpha_68']['weight']
            deltawindow = PARAMETER['alpha_68']['deltawindow']
            return Df_Int((ts_rank(correlation(rank(df_high), rank(df_adv(15)), corrwindow), rankwindow) < rank(delta(((df_close * weight) + (df_low * (1 - weight))), deltawindow)))) * (-1)
            
        def Alpha69():
            '''
            Formula:((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),
4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),
9.0615)) * -1)
            '''
            corrwindow = PARAMETER['alpha_69']['corrwindow']
            rankwindow = PARAMETER['alpha_69']['rankwindow']
            weight = PARAMETER['alpha_69']['weight']
            deltawindow = PARAMETER['alpha_69']['deltawindow']
            maxwindow = PARAMETER['alpha_69']['maxwindow']         
            return (rank(ts_max(delta(IndNeutralize(df_vwap, df_ind['Industry']), deltawindow),maxwindow))**ts_rank(correlation(((df_close * weight) + (df_vwap * (1 - weight))), df_adv(20), corrwindow),rankwindow)) * (-1)
            
        def Alpha70():
            '''
            Formula:((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,
IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
            '''
            corrwindow = PARAMETER['alpha_70']['corrwindow']
            rankwindow = PARAMETER['alpha_70']['rankwindow']
            deltawindow = PARAMETER['alpha_70']['deltawindow']
            return (rank(delta(df_vwap, deltawindow))**ts_rank(correlation(IndNeutralize(df_close,df_ind['Industry']), df_adv(50), corrwindow), rankwindow)) * (-1)
        
        def Alpha71():#dodododododododo
            '''
            Formula:max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,
12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +
vwap)))^2), 16.4662), 4.4388))
            '''
            corrwindow = PARAMETER['alpha_71']['corrwindow']
            rankwindow1 = PARAMETER['alpha_71']['rankwindow1']
            rankwindow2 = PARAMETER['alpha_71']['rankwindow2']
            rankwindow3 = PARAMETER['alpha_71']['rankwindow3']
            rankwindow4 = PARAMETER['alpha_71']['rankwindow4']
            decaywindow1 = PARAMETER['alpha_71']['decaywindow1']
            decaywindow2 = PARAMETER['alpha_71']['decaywindow2']
            return max(ts_rank(decay_linear(correlation(ts_rank(df_close, rankwindow1), ts_rank(df_adv(180),rankwindow2), corrwindow), decaywindow1), rankwindow3), ts_rank(decay_linear((rank(((df_low + df_open) - (df_vwap + df_vwap))).pow(2)), decaywindow2), rankwindow4))
            
        def Alpha72():#dodododododododo
            '''
            Formula:(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /
rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),
2.95011)))
            '''
            corrwindow1 = PARAMETER['alpha_72']['corrwindow1']
            corrwindow2 = PARAMETER['alpha_72']['corrwindow2']
            rankwindow1 = PARAMETER['alpha_72']['rankwindow1']
            rankwindow2 = PARAMETER['alpha_72']['rankwindow2']
            decaywindow1 = PARAMETER['alpha_72']['decaywindow1']
            decaywindow2 = PARAMETER['alpha_72']['decaywindow2']
            return (rank(decay_linear(correlation(((df_high + df_low) / 2), df_adv(40), corrwindow1), decaywindow1)) / rank(decay_linear(correlation(ts_rank(df_vwap, rankwindow1), ts_rank(df_volume, rankwindow2), corrwindow2), decaywindow2)))

        def Alpha73():
            '''
            Formula:(max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *
0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
            '''
            rankwindow = PARAMETER['alpha_73']['rankwindow']
            weight = PARAMETER['alpha_73']['weight']
            decaywindow1 = PARAMETER['alpha_73']['decaywindow1']
            decaywindow2 = PARAMETER['alpha_73']['decaywindow2']
            deltawindow1 = PARAMETER['alpha_73']['deltawindow1']
            deltawindow2 = PARAMETER['alpha_73']['deltawindow2']
            return max(rank(decay_linear(delta(df_vwap, deltawindow1), decaywindow1)),ts_rank(decay_linear(((delta(((df_open * weight) + (df_low * (1 - weight))), deltawindow2) / ((df_open *weight) + (df_low * (1 - weight)))) * (-1)), decaywindow2), rankwindow)) * (-1)

        def Alpha74():
            '''
            Formula:((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <
rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
            '''
            corrwindow1 = PARAMETER['alpha_74']['corrwindow1']
            corrwindow2 = PARAMETER['alpha_74']['corrwindow2']
            sumwindow = PARAMETER['alpha_74']['sumwindow']
            weight = PARAMETER['alpha_74']['weight']
            return Df_Int((rank(correlation(df_close, ts_sum(df_adv(30), sumwindow),corrwindow1)) < rank(correlation(rank(((df_high * weight) + (df_vwap * (1 - weight)))), rank(df_volume), corrwindow2))))* (-1)
            
        def Alpha75():
            '''
            Formula:(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),
12.4413)))
            '''
            corrwindow1 = PARAMETER['alpha_75']['corrwindow1']
            corrwindow2 = PARAMETER['alpha_75']['corrwindow2']
            return Df_Int((rank(correlation(df_vwap, df_volume, corrwindow1)) < rank(correlation(rank(df_low), rank(df_adv(50)),corrwindow2))))
            
        def Alpha76():
            '''
            Formula:(max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,
8.14941), 19.569), 17.1543), 19.383)) * -1)
            '''
            rankwindow1 = PARAMETER['alpha_76']['rankwindow1']
            rankwindow2 = PARAMETER['alpha_76']['rankwindow2']
            decaywindow1 = PARAMETER['alpha_76']['decaywindow1']
            decaywindow2 = PARAMETER['alpha_76']['decaywindow2']
            corrwindow = PARAMETER['alpha_76']['corrwindow']
            deltawindow = PARAMETER['alpha_76']['deltawindow']
            return max(rank(decay_linear(delta(df_vwap, deltawindow), decaywindow1)),ts_rank(decay_linear(ts_rank(correlation(IndNeutralize(df_low, df_ind['Sector']), df_adv(81),corrwindow), rankwindow1), decaywindow2), rankwindow2)) * (-1)
        
        def Alpha77():
            '''
            Formula:min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
            '''
            decaywindow1 = PARAMETER['alpha_77']['decaywindow1']
            decaywindow2 = PARAMETER['alpha_77']['decaywindow2']
            corrwindow = PARAMETER['alpha_77']['corrwindow']
            rank1 = rank(decay_linear(((((df_high + df_low) * 0.5) + df_high) - (df_vwap + df_high)), decaywindow1))
            rank2 = rank(decay_linear(correlation(((df_high + df_low) * 0.5), df_adv(40), corrwindow),decaywindow2))
            return min(rank1,rank2)
            
        def Alpha78():
            '''
            Formula:(rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
            '''
            sumwindow = PARAMETER['alpha_78']['sumwindow']
            weight = PARAMETER['alpha_78']['weight']
            corrwindow1 = PARAMETER['alpha_78']['corrwindow1']
            corrwindow2 = PARAMETER['alpha_78']['corrwindow2']
            return (rank(correlation(ts_sum(((df_low * weight) + (df_vwap * (1 - weight))), sumwindow),ts_sum(df_adv(40), sumwindow),corrwindow1))**rank(correlation(rank(df_vwap), rank(df_volume), corrwindow2)))
            
            
        def Alpha79():
            '''
            Formula:(rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),
IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,
9.18637), 14.6644)))
            '''
            rankwindow1 = PARAMETER['alpha_79']['rankwindow1']
            rankwindow2 = PARAMETER['alpha_79']['rankwindow2']
            corrwindow = PARAMETER['alpha_79']['corrwindow']
            deltawindow = PARAMETER['alpha_79']['deltawindow']
            weight = PARAMETER['alpha_79']['weight']
            return Df_Int(rank(delta(IndNeutralize(((df_close * weight) + (df_open * (1 - weight))),df_ind['Sector']), deltawindow)) < rank(correlation(ts_rank(df_vwap, rankwindow1), ts_rank(df_adv(150),rankwindow2), corrwindow)))
        
        def Alpha80():
            '''
            Formula:((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),
IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
            '''
            corrwindow = PARAMETER['alpha_80']['corrwindow']
            rankwindow = PARAMETER['alpha_80']['rankwindow']
            weight = PARAMETER['alpha_80']['weight']
            deltawindow = PARAMETER['alpha_80']['deltawindow']
            return (rank(sign(delta(IndNeutralize(((df_open * weight) + (df_high * (1 - weight))),df_ind['Industry']), deltawindow)))**ts_rank(correlation(df_high, df_adv(10), corrwindow), rankwindow)) * (-1)


        def Alpha81():
            ''' 
            Formula: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_81']

            sum1 = ts_sum(df_adv(10), int(_param['sum_window']))
            rank_correlation1 = rank(correlation(
                df_vwap, sum1, int(_param['corr_volume_window'])))
            # rank1__4=rank_correlation1^4
            rank1__4 = pow(rank_correlation1, int(_param['rank_power']))
            rank2 = rank((rank1__4))
            prod1 = product(rank2, int(_param['prod_window']))
            correlation2 = correlation(rank(df_vwap),  rank(
                df_volume), int(_param['corr_rankvolume_window']))
            rank_cmp = (rank(Log(prod1)) < rank(correlation2))
            return (rank_cmp * _param['alpha_sign'])
            pass

        def Alpha82():
            ''' 
            Formula: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), ts_rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_82']

            delta_1 = delta(df_open, int(_param['delta_step']))
            decay_1 = decay_linear(delta_1, int(_param['ldecay_window']))
            indNT1=IndNeutralize(df_volume, df_ind['Sector'])
            correlation_1 = correlation(indNT1, ((
                df_open * _param['open_scale']) + (df_open * (1 - _param['open_scale']))), int(_param['corr_window']))
            decay_2 = decay_linear(
                correlation_1, int(_param['ldecay_2_window']))
            tsrank_1 = ts_rank(decay_2, int(_param['tsrank_window']))
            rank_1 = rank(decay_1)
            min_1 = Min(rank_1, tsrank_1)
            return (min_1 * _param['alpha_sign'])
            pass

        def Alpha83():
            ''' 
            Formula: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_83']

            sum_1 = ts_sum(df_close, int(
                _param['sum_window'])) / int(_param['sum_window'])
            delay_1 = delay(((df_high - df_low) / (sum_1)),
                            int(_param['delay_step']))
            nominator_1 = rank(delay_1) * rank(rank(df_volume))
            nominator_2 = (df_high - df_low) / (sum_1)
            denominator_1 = nominator_2 / (df_vwap - df_close)
            return (nominator_1 / denominator_1)
            pass

        def Alpha84():
            ''' 
            Formula: SignedPower(ts_rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_84']

            tsmax_1 = ts_max(df_vwap, int(_param['tsmax_window']))
            tsrank_1 = ts_rank((df_vwap - tsmax_1),
                               int(_param['tsrank_window']))
            delta_1 = delta(df_close, int(_param['delta_step']))
            return pow(tsrank_1, delta_1)
            pass

        def Alpha85():
            ''' 
            Formula: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(ts_rank(((high + low) / 2), 3.70596), ts_rank(volume, 10.1595), 7.11408)))
            Explanation: 
                
            '''

            _param = PARAMETER['alpha_85']

            correlation_1 = correlation(((df_high * _param['high_pct']) + (df_close * (
                1 - _param['high_pct']))), df_adv(30), int(_param['corr_window_1']))
            tsrank_1 = ts_rank(((df_high + df_low) / 2),
                               int(_param['tsrank_highlow_window']))
            tsrank_2 = ts_rank(df_volume, int(_param['tsrank_volume_window']))
            correlation_2 = correlation(
                tsrank_1, tsrank_2, int(_param['corr_window_2']))
            # return ( rank(correlation_1)^ rank(correlation_2))
            return pow(rank(correlation_1), rank(correlation_2))

        def Alpha86():
            ''' 
            Formula: ((ts_rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_86']

            sum_1 = ts_sum(df_adv(20), int(_param['sum_window']))
            correlation_1 = correlation(
                df_close, sum_1, int(_param['corr_window']))
            tsrank_1 = ts_rank(
                correlation_1, int(_param['tsrank_window']))
            rank_1 = rank(((df_open + df_close) - (df_vwap + df_open)))
            return ((tsrank_1 < rank_1) * _param['alpha_sign'])

        def Alpha87():
            ''' 
            Formula: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), ts_rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_87']
            df_adv81 = df_volume.rolling(
                window=int(_param['adv_window'])).mean()

            delta_1 = delta(((df_close * _param['close_pct']) + (
                df_vwap * (1 - _param['close_pct']))), int(_param['delta_step']))
            decay_1 = decay_linear(delta_1, int(_param['ldecay_window_1']))
            correlation_1 = correlation(IndNeutralize(
                df_adv81, df_ind['Industry']), df_close, int(_param['corr_window']))
            decay_2 = decay_linear(np.abs(correlation_1),
                                   int(_param['ldecay_window_2']))
            tsrank_1 = ts_rank(decay_2, int(_param['tsrank_window']))
            return (Max(rank(decay_1), tsrank_1) * _param['alpha_sign'])

        def Alpha88():
            ''' 
            Formula: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), ts_rank(decay_linear(correlation(ts_rank(close, 8.44728), ts_rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_88']

            rank_1 = ((rank(df_open) + rank(df_low)) -
                      (rank(df_high) + rank(df_close)))
            decay_1 = decay_linear(rank_1, int(_param['ldecay_window_1']))
            tsrank_1 = ts_rank(df_close, int(_param['tsrank_window_1']))
            tsrank_2 = ts_rank(df_adv(60), int(_param['tsrank_window_2']))
            correlation_1 = correlation(
                tsrank_1, tsrank_2, int(_param['corr_window']))
            decay_2 = decay_linear(
                correlation_1, int(_param['ldecay_window_2']))
            tsrank_3 = ts_rank(decay_2, int(_param['tsrank_window_3']))
            return Min(rank(decay_1), tsrank_3)

        def Alpha89():
            ''' 
            Formula: (ts_rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - ts_rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_89']

            correlation_1 = correlation(((df_low * _param['low_pct']) + (
                df_low * (1 - _param['low_pct']))), df_adv(10), int(_param['corr_window']))
            decay_1 = decay_linear(correlation_1, int(_param['ldecay_window']))
            tsrank_1 = ts_rank(decay_1, int(_param['tsrank_window']))
            delta_1 = delta(IndNeutralize(
                df_vwap, df_ind['Industry']), int(_param['delta_step']))
            decay_2 = decay_linear(delta_1, int(_param['ldecay_window_2']))
            tsrank_2 = ts_rank(decay_2, int(_param['tsrank_window_2']))
            return (tsrank_1 - tsrank_2)

        def Alpha90():
            ''' 
            Formula: ((rank((close - ts_max(close, 4.66719)))^ts_rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
            Explanation: 
                
            '''

            _param = PARAMETER['alpha_90']

            rank1 = rank((df_close - ts_max(df_close, int(_param['tsmax_window']))))
            indNT1=IndNeutralize(df_adv(40), df_ind['Subindustry'])
            correlation_1 = correlation(indNT1, df_low, int(_param['corr_window']))
            tsrank1 = ts_rank(correlation_1, int(_param['tsrank_window']))
            return (pow(rank1, tsrank1) * _param['alpha_sign'])

        def Alpha91():
            ''' 
            Formula: ((ts_rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_91']

            correlation1 = correlation(IndNeutralize(
                df_close, df_ind['Industry']), df_volume, int(_param['corr_window']))
            decay1 = decay_linear(correlation1, int(_param['ldecay_window']))
            decay2 = decay_linear(decay1, int(_param['ldecay_window_2']))
            tsrank1 = ts_rank(decay2, int(_param['tsrank_window']))
            correlation2 = correlation(df_vwap, df_adv(
                30), int(_param['corr_window_2']))
            decay3 = decay_linear(correlation2, int(_param['ldecay_window_3']))
            return ((tsrank1 - rank(decay3)) * _param['alpha_sign'])

        def Alpha92():
            ''' 
            Formula: 
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_92']

            decay1 = decay_linear(((((df_high + df_low) / 2) + df_close)
                                   < (df_low + df_open)), int(_param['ldecay_window_1']))
            tsrank1 = ts_rank(decay1, int(_param['tsrank_window_1']))
            correlation1 = correlation(rank(df_low),  rank(
                df_adv(30)), int(_param['corr_window']))
            decay2 = decay_linear(correlation1, int(_param['ldecay_window_2']))
            tsrank2 = ts_rank(decay2, int(_param['tsrank_window_2']))
            return Min(tsrank1, tsrank2)

        def Alpha93():
            ''' 
            Formula: (ts_rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_93']
            df_adv81 = df_volume.rolling(
                window=int(_param['adv_window'])).mean()

            correlation1 = correlation(IndNeutralize(
                df_vwap, df_ind['Industry']), df_adv81, int(_param['corr_window']))
            decay1 = decay_linear(correlation1, int(_param['ldecay_window_1']))
            tsrank1 = ts_rank(decay1, int(_param['tsrank_window']))
            delta1 = delta(((df_close * _param['close_pct']) + (
                df_vwap * (1 - _param['close_pct']))), int(_param['delta_step']))
            decay2 = decay_linear(delta1, int(_param['ldecay_window_2']))
            return (tsrank1 / rank(decay2))

        def Alpha94():
            ''' 
            Formula:  ((rank((vwap - ts_min(vwap, 11.5783)))^ts_rank(correlation(ts_rank(vwap, 19.6462), ts_rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_94']

            tsmin = ts_min(df_vwap,  int(_param['tsmin_window']))
            rank_1 = rank((df_vwap - tsmin))
            tsrank_1 = ts_rank(df_vwap, int(_param['tsrank_window_1']))
            tsrank_2 = ts_rank(df_adv(60), int(_param['tsrank_window_2']))
            correlation_1 = correlation(
                tsrank_1, tsrank_2, int(_param['corr_window']))
            tsrank_3 = ts_rank(correlation_1, int(_param['tsrank_window_3']))
            return (pow(rank_1, tsrank_3) * _param['alpha_sign'])

        def Alpha95():
            ''' 
            Formula: (rank((open - ts_min(open, 12.4105))) < ts_rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_95']

            tsmin = ts_min(df_open, int(_param['tsmin_window']))
            rank_1 = rank((df_open - tsmin))
            sum_1 = ts_sum(((df_high + df_low) / 2), int(_param['sum_window']))
            sum_2 = ts_sum(df_adv(40), int(_param['sum_window']))
            correlation_1 = correlation(
                sum_1, sum_2, int(_param['corr_window']))
            rank_2 = rank(correlation_1)
            tsrank_1 = ts_rank(
                pow(rank_2, _param['power_times']), int(_param['tsrank_window']))
            return (rank_1 < tsrank_1) * 1

        def Alpha96():
            ''' 
            Formula: (max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), ts_rank(decay_linear(ts_argMax(correlation(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_96']

            correlation_1 = correlation(rank(df_vwap),  rank(
                df_volume), int(_param['corr_window_1']))
            decay_1 = decay_linear(
                correlation_1, int(_param['ldecay_window_1']))
            tsrank_1 = ts_rank(decay_1, int(_param['tsrank_window_1']))
            tsrank_2 = ts_rank(df_close, int(_param['tsrank_window_2']))
            tsrank_3 = ts_rank(df_adv(60), int(_param['tsrank_window_3']))
            correlation_2 = correlation(
                tsrank_2, tsrank_3, int(_param['corr_window_2']))
            tsargmax_1 = ts_argmax(
                correlation_2, int(_param['tsargmax_window']))
            decay_2 = decay_linear(tsargmax_1, int(_param['ldecay_window_2']))
            tsrank_4 = ts_rank(decay_2, int(_param['tsrank_window_4']))
            return (Max(tsrank_1, tsrank_4) * _param['alpha_sign'])

        def Alpha97():
            ''' 
            Formula: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 7.87871), ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
            Explanation: 
               
            '''
            _param = PARAMETER['alpha_97']

            delta_1 = delta(IndNeutralize(((df_low * _param['low_pct']) + (df_vwap * (
                1 - _param['low_pct']))), df_ind['Industry']), int(_param['delta_window']))
            decay_1 = decay_linear(delta_1, int(_param['ldecay_window_1']))
            tsrank_1 = ts_rank(df_low, int(_param['tsrank_window_1']))
            tsrank_2 = ts_rank(df_adv(60), int(_param['tsrank_window_2']))
            correlation_1 = correlation(
                tsrank_1, tsrank_2, int(_param['corr_window']))
            tsrank_3 = ts_rank(correlation_1, int(_param['tsrank_window_3']))
            decay_2 = decay_linear(tsrank_3, int(_param['ldecay_window_2']))
            tsrank_4 = ts_rank(decay_2, int(_param['tsrank_window_4']))
            return ((rank(decay_1) - tsrank_4) * _param['alpha_sign'])

        def Alpha98():
            ''' 
            Formula: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(ts_rank(ts_argMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_98']
            df_adv5 = df_volume.rolling(
                window=int(_param['adv_window'])).mean()

            sum_1 = ts_sum(df_adv5, int(_param['sum_window']))
            correlation_1 = correlation(
                df_vwap, sum_1, int(_param['corr_window']))
            decay_1 = decay_linear(
                correlation_1, int(_param['ldecay_window_1']))
            correlation_2 = correlation(rank(df_open),  rank(
                df_adv(15)), int(_param['corr_window_2']))
            tsargmin_1 = ts_argmin(
                correlation_2, int(_param['tsargmin_window']))
            tsrank_1 = ts_rank(tsargmin_1, int(_param['tsrank_window_1']))
            decay_2 = decay_linear(tsrank_1, int(_param['ldecay_window_2']))
            return (rank(decay_1) - rank(decay_2))

        def Alpha99():
            ''' 
            Formula:  ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_99']

            sum_1 = ts_sum(((df_high + df_low) / 2), int(_param['sum_window']))
            sum_2 = ts_sum(df_adv(60), int(_param['sum_window']))
            correlation_1 = correlation(
                sum_1, sum_2, int(_param['corr_window_1']))
            correlation_2 = correlation(
                df_low, df_volume, int(_param['corr_window_2']))
            return ((rank(correlation_1) < rank(correlation_2)) * _param['alpha_sign'])

        def Alpha100():
            ''' 
            Formula: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_100']

            correlation_1 = correlation(df_close,  rank(
                df_adv(20)), int(_param['corr_window']))
            tsargmin_1 = ts_argmin(df_close, int(_param['tsargmin_window']))
            rank_1 = rank(
                ((((df_close - df_low) - (df_high - df_close)) / (df_high - df_low)) * df_volume))
            scale_1 = scale(IndNeutralize(IndNeutralize(
                rank_1, df_ind['Subindustry']), df_ind['Subindustry']))
            scale_2 = scale(IndNeutralize(
                (correlation_1 - rank(tsargmin_1)), df_ind['Subindustry']))
            return (_param['threshold'] - (_param['scale_2'] * (((_param['scale_1'] * scale_1) - scale_2) * (df_volume / df_adv(20)))))

        def Alpha101():
            ''' 
            Formula:  ((close - open) / ((high - low) + .001)) 
            Explanation: 
                
            '''
            _param = PARAMETER['alpha_101']

            return ((df_close - df_open) / (df_high - df_low + 0.001))
   
 
        #Alpha function map: More function will be implemented and added into this map
        function = {1 : Alpha1,
                2 : Alpha2,
                3 : Alpha3,
                4 : Alpha4,
                5 : Alpha5,
                6 : Alpha6,
                7 : Alpha7,
                8 : Alpha8,
                9 : Alpha9,
                10 : Alpha10,
                11 : Alpha11,
                12 : Alpha12,
                13 : Alpha13,
                14 : Alpha14,
                15 : Alpha15,
                16 : Alpha16,
                17 : Alpha17,
                18 : Alpha18,
                19 : Alpha19,
                20 : Alpha20,
                21 : Alpha21,
                22 : Alpha22,
                23 : Alpha23,
                24 : Alpha24,
                25 : Alpha25,
                26 : Alpha26,
                27 : Alpha27,
                28 : Alpha28,
                29 : Alpha29,
                30 : Alpha30,
                31 : Alpha31,
                32 : Alpha32,
                33 : Alpha33,
                34 : Alpha34,
                35 : Alpha35,
                36 : Alpha36,
                37 : Alpha37,
                38 : Alpha38,
                39 : Alpha39,
                40 : Alpha40,
                41 : Alpha41,
                42 : Alpha42,
                43 : Alpha43,
                44 : Alpha44,
                45 : Alpha45,
                46 : Alpha46,
                47 : Alpha47,
                48 : Alpha48,
                49 : Alpha49,
                50 : Alpha50,
                51 : Alpha51,
                52 : Alpha52,
                53 : Alpha53,
                54 : Alpha54,
                55 : Alpha55,
                56 : Alpha56,
                57 : Alpha57,
                58 : Alpha58,
                59 : Alpha59,
                60 : Alpha60,
                61 : Alpha61,
                62 : Alpha62,
                63 : Alpha63,
                64 : Alpha64,
                65 : Alpha65,
                66 : Alpha66,
                67 : Alpha67,
                68 : Alpha68,
                69 : Alpha69,
                70 : Alpha70,
                71 : Alpha71,
                72 : Alpha72,
                73 : Alpha73,
                74 : Alpha74,
                75 : Alpha75,
                76 : Alpha76,
                77 : Alpha77,
                78 : Alpha78,
                79 : Alpha79,
                80 : Alpha80,
                81 : Alpha81,
                82 : Alpha82,
                83 : Alpha83,
                84 : Alpha84,
                85 : Alpha85,
                86 : Alpha86,
                87 : Alpha87,
                88 : Alpha88,
                89 : Alpha89,
                90 : Alpha90,
                91 : Alpha91,
                92 : Alpha92,
                93 : Alpha93,
                94 : Alpha94,
                95 : Alpha95,
                96 : Alpha96,
                97 : Alpha97,
                98 : Alpha98,
                99 : Alpha99,
                100 : Alpha100,
                101 : Alpha101
                }

        #Calculate Alpha
        if isinstance(Alpha, int):
            return function[Alpha]()
        else:
            print("Alpha option is not correct. Please check")

