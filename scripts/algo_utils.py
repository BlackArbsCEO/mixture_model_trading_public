

import pandas as pd
import numpy as np
from math import ceil, floor
import scipy.stats as stats
import sklearn.mixture as mix
from datetime import datetime, timedelta
import time
import decimal as d
import json 

#######################################################################
# gmm functions
#######################################################################


def make_gmm(n_components=None, max_iter=150, random_state=None):
    """fn: create gmm object"""
    model_kwds = dict(n_components=n_components, 
                      max_iter=max_iter,
                      n_init=100,
                      init_params='random',
                      random_state=random_state)

    gmm = mix.GaussianMixture(**model_kwds)
    return gmm
    
def make_returns(df):
    return np.log(df/df.shift(1)).dropna()
    
#######################################################################
# pred df functions
#######################################################################    
def in_range(df):
    """fn: add binary column for predictions within CI"""
    wins = df.query("low_ci < current_return < high_ci").index
    in_list = [1 if i in wins else 0 for i in df.index]
    df = df.assign(in_range=in_list)
    return df

def get_state_prob(df):
    state_prob = []
    for row in df[['ith_state','last_prob_class_0', 'last_prob_class_1']].itertuples():
        prob = pd.eval(f'row.last_prob_class_{row.ith_state}')
        state_prob.append(prob)
    return df.assign(state_prob=state_prob)
    
def get_outlier_direction(df):
    """"""
    direction = []
    for row in df[['high_ci', 'current_return']].itertuples(index=False):
        if row[-1] > row[0]: # actual_returns > high_ci
            direction.append('too_high')
        else: 
            direction.append('too_low')
    df = df.assign(direction=direction)
    return df

def buys(df, thres=0.5):
    buys = []
    for row in df.itertuples():
        if (row.ith_state==0 
            and row.mu_diff>0 
            and row.in_range==1
            and row.state_prob>thres
            and row.direction=='too_low'):
            buys.append(1)
        elif (row.ith_state==1 
              and row.mu_diff <0
              and row.in_range==1
              and row.state_prob>thres              
              and row.direction=='too_low'):
            buys.append(1)
        else:
            buys.append(0)
    return df.assign(buys=buys) 
    
def make_final_pred_df(pred_rows, cols, thres, sym):
    pred_df = (pd.DataFrame(pred_rows, columns=cols)
               .assign(mu_diff=lambda df: df.avg_class_0_mean-df.avg_class_1_mean)
               .assign(std_diff=lambda df: df.avg_class_0_std-df.avg_class_1_std)
               .pipe(in_range)
               .pipe(get_state_prob)
               .pipe(get_outlier_direction)
               .pipe(buys, thres=thres)
               .set_index('Dates')
               .assign(Dates = lambda df: df.index))
    return pred_df    
    
#######################################################################
# updating historical timeseries dataframes
#######################################################################

def how_many_days(current_date, most_recent_date):
    """compute how many days to request from history api
    # args: both are datetime objects 
    """    
    return (current_date - most_recent_date).days
    
def zero_days_to_request(days_to_request): 
    """check if days to request is equal to 0 
       if yes exit algorithm
    """
    # request only days that are missing from our dataset
    if days_to_request==0:
        return True
        
def make_update_df(old, new, lookback):
    """combines and cleans numeric timeseries dataframes
       for updates
   
    # args
        old, new: pandas dataframes
        lookback: numeric 
        
    # returns
        both: combined dataframe 
    """
    # combine datasets                      
    both = pd.concat([old, new]) 
    # clean it up and keep only lookback period
    return (both
            .drop_duplicates()
            .sort_index()
            .iloc[-lookback:]) 
     
#######################################################################
# order execution functions
#######################################################################
def get_open_order_secs(open_orders):
    """func to return list of symbols
        if open order list is populated
    """
    if open_orders: # if list is populated
        open_order_secs = [order.Symbol for order in open_orders]
    else: 
        open_order_secs = []
    return open_order_secs