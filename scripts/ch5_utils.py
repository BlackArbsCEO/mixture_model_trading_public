# import standard libs
from pathlib import PurePath, Path
import sys
import time
from collections import OrderedDict as od
import re
import os
import json

# import python scientific stack
import pandas as pd
import pandas_datareader.data as web
pd.set_option('display.max_rows', 50)
import numpy as np
import sklearn.mixture as mix
import scipy.stats as stats
import math
import pymc3 as pm
from theano import shared, theano as tt

# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# format simulated bayesian paths
# --------------------------------------------------

class pymc3_helper:
    """class of convenience funcs for pymc3 analysis"""
    def __init__(self):
        pass
    
    def strip_derived_rvs(self, rvs):
        '''Convenience fn: remove PyMC3-generated RVs from a list

        Code adapted from:
            https://github.com/jonsedar/pymc3_vs_pystan/blob/master/convenience_functions.py
        '''
        ret_rvs = []
        for rv in rvs:
            if not (re.search('_log',rv.name) or re.search('_interval',rv.name)):
                ret_rvs.append(rv)     
        return ret_rvs


    def plot_traces_pymc(self, trcs, varnames=None):
        ''' Convenience fn: plot traces with overlaid means and values
        Code adapted from:
            https://github.com/jonsedar/pymc3_vs_pystan/blob/master/convenience_functions.py
        '''        

        nrows = len(trcs.varnames)
        if varnames is not None:
            nrows = len(varnames)

        ax = pm.traceplot(trcs, varnames=varnames, figsize=(12,nrows*1.4),
                          lines={k: v['mean'] for k, v in 
                                 pm.summary(trcs,varnames=varnames).iterrows()})

        for i, mn in enumerate(pm.summary(trcs, varnames=varnames)['mean']):
            ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data',
                             xytext=(5,10), textcoords='offset points', rotation=90,
                             va='bottom', fontsize='large', color='#AA0022')

    def get_model_varnames(self, model):
        return [rv.name for rv in self.strip_derived_rvs(model.unobserved_RVs)]            
    # --------------------------------------------------
    # tick labeling for pymc3 compareplot

    def extract_yticks(self, ticks):
        """fn: extract yticks from pm.compareplot"""
        t = [ticks[x].get_text() for x in range(len(ticks))]
        return [int(x)  if x != '' else '' for x in t]

    def make_new_ticklabels(self, LABEL_MAP, ticks):
        """fn: make new ticklabels with model names"""
        t = self.extract_yticks(ticks)
        new_ticklabels = [LABEL_MAP[x] if x != '' else x for x in t]
        return new_ticklabels     

    # --------------------------------------------------
    # format simulated bayesian paths

    def format_sim_path_df(self, sim_paths, starting_equity):
        """fn: format simulated bayesian path df
        
        # args
            sim_paths: df, simulated portfolio equity paths
            starting_equity: int, portfolio starting cash
        # returns
            df_update: df, formatted df
        """
        df_update = (sim_paths.cumsum().add(1).multiply(starting_equity))
        sim_start_date = df_update.index.min() - pd.Timedelta('1D')
        df_update.loc[sim_start_date, :] = [starting_equity]*len(df_update.columns)
        df_update.sort_index(inplace=True)
        return df_update

    # --------------------------------------------------
    # helper funcs for simulating bayesian paths  
    def make_mean_path(self, sim_path_df):
        return (sim_path_df
                .assign(mean_sim_path=lambda df: df.mean(1))) 

    def make_ppc_df(self, ppc_samples):
        """fn: to make ppc df from ppc sample output"""

        ppc = self.make_mean_path(ppc_samples) # add mean path to df
        #ppc = ppc['mean_sim_path'].to_frame()
        return ppc

    def get_samp_cols(self, df, n_cols):
        """fn: to extract subset of df columns and glue to mean simulated portfolio"""
        return np.hstack((df.columns[:n_cols], np.array(['mean_sim_path'])))

    def _make_ppc_cuml_df(self, ppc, n_cols=2000):
        """fn: to compute df of cuml sum of ppc sampled returns"""
        cols = self.get_samp_cols(ppc, n_cols)
        cuml_df = ppc.loc[:,cols].cumsum()
        return cuml_df

    def make_df(self, ppc_samples, n_cols=2000):
        ppc = self.make_ppc_df(ppc_samples)
        cuml_df = self._make_ppc_cuml_df(ppc, n_cols)
        return ppc, cuml_df

    # --------------------------------------------------
    # helper funcs for evaluating predicted portfolio ending values

    def get_paths(self, ppc, n_cols):
        cols = self.get_samp_cols(ppc, n_cols)
        sim_path_df = self.format_sim_path_df(ppc.loc[:,cols],100_000)
        return sim_path_df

    def get_end_vals(self, path_df):
        return path_df.iloc[-1][path_df.iloc[-1] > 0]    
    # --------------------------------------------------
    # helper funcs for plotting predicted portfolio ending values

    def update_xlabels(self, ax):
        xlabels = ['${:0,.0f}'.format(label)
                   for label in ax.get_xticks()]
        ax.set_xticklabels(xlabels, rotation=30)
    
    def plot_port_end_dist(self, sim_end_val, axes,
                           quantiles=[0.05,.5,.95],
                           model_name=None):
        """fn: plot simulated portfolio ending values distributions
        """
        if not model_name:
            model_name = 'Model'
            
        qtiles = sim_end_val.quantile(quantiles)
        colors=['r','k','g']
        COLOR_LABEL_MAP = od(r=f'5% = ${qtiles[0.05]:,.2f}',
                             k=f'50% = ${qtiles[0.50]:,.2f}',
                             g=f'95% = ${qtiles[0.95]:,.2f}')

        # plots
        sns.distplot(sim_end_val.values, ax=axes[0], 
                     kde_kws={'cumulative':False, 'shade':True})
        sns.distplot(sim_end_val.values, ax=axes[1],
                     kde_kws={'cumulative':True, 'shade':True})

        # make legends
        lines = []
        for k, v in COLOR_LABEL_MAP.items():
            tmp_line = mpl.lines.Line2D([], [],
                                        color=k, label=v,
                                        linestyle='-.')
            lines.append(tmp_line)

        # plot vlines and legends
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            ax.vlines(qtiles.values, ymin, ymax, 
                      colors=colors, linestyle='-.')  
            ax.legend(loc='upper left', handles=lines)
            self.update_xlabels(ax)

        axes[1].set_ylim(0,1)
        plt.suptitle(f'{model_name} || Distribution of Simulated Portfolio Ending Values',
                     fontsize=14);

    def plot_cagr_bar(self, ser, quantiles, ax, model_name=None):
        
        if not model_name: model_name = 'Model'
            
        ser[quantiles].plot.bar(ax=ax)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin-0.01, ymax+0.01))
        ax.axhline(0, color='k', lw=1.)

        # bar plot label solution found here:
        #  https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart#28931750
        rects = ax.patches

        # For each bar: Place a label
        for rect in rects:
            # Get X and Y placement of label from rect
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = 5
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1%}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va,                      # Vertically align label differently for
                fontsize=16,)               # positive and negative values.
            ax.set_xlabel('Quantiles')
            ax.set_ylabel('CAGR')
            ax.set_title(f'{model_name} || Simulated CAGR Quantile Values')                             
