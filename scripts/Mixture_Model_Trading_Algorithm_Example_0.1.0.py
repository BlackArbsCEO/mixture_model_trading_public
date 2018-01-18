from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Indicators")
AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *

import pandas as pd
import numpy as np
from math import ceil, floor
import scipy.stats as stats
import sklearn.mixture as mix
from datetime import datetime, timedelta
import time
import json

# ------------------------------------------------------------------------------
# setup parameter registry 
# ------------------------------------------------------------------------------

PARAMETER_REGISTRY = {}

def register_param(name, value):
    PARAMETER_REGISTRY[name] = value
    return value

# ------------------------------------------------------------------------------
# global variables and parameters
# ------------------------------------------------------------------------------

### strategy information ###
is_long_only = register_param('is_long_only', True)
N_SAMPLES = register_param('n samples (bootstrapping distr.)', 1000)

### choose distribution ###
sample_distr = register_param('sampling distr', 'normal distribution')
#sample_distr = register_param('sampling distr', 'laplace')
#sample_distr = register_param('sampling distr.', 'johnsonsu')
### if using jsu register a, b parameters ###
#a, b = register_param('a (jsu)', 0.2), register_param('b (jsu)', 0.9)

### gmm init variables ###
RANDOM_STATE = register_param('random state', 777)
ALPHA = register_param('alpha', 0.95) # for sampling confidence intervals
N_COMPONENTS = register_param('n components (GMM)', 4)
MAX_ITER = register_param('max iterations (GMM)', 100)
N_INIT = register_param('n inits (GMM)', 25)

# ------------------------------------------------------------------------------
# global funcs
# ------------------------------------------------------------------------------

def make_gmm(n_components=N_COMPONENTS, max_iter=MAX_ITER, 
             n_init=N_INIT, random_state=RANDOM_STATE):
    """fn: create gmm object"""
    model_kwds = dict(n_components=n_components, 
                      max_iter=max_iter,
                      n_init=n_init,
                      init_params='random',
                      random_state=random_state)

    gmm = mix.GaussianMixture(**model_kwds)
    return gmm
    
def make_returns(df):
    return np.log(df/df.shift(1)).dropna()
        
# ------------------------------------------------------------------------------
# algorithm 
# ------------------------------------------------------------------------------

class TradingWithGMM(QCAlgorithm):
    """Algorithm which implements GMM framework"""
        
    def Initialize(self):
        '''All algorithms must initialized.'''

        self.SetStartDate(2007,1,1)  #Set Start Date
        self.SetEndDate(2017,12,31)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        
        # -----------------------------------------------------------------------------
        # init brokerage model, important for realistic slippage/commission modeling
        # especially important if using leverage which requires margin account
        # -----------------------------------------------------------------------------
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage,
                               AccountType.Margin)
                               
        # -----------------------------------------------------------------------------
        # init custom universe
        # -----------------------------------------------------------------------------
        symbol_list = ["SPY", "QQQ", "DIA", "EFA", "EEM",  "TLT", 'AGG', 'LQD', "GLD"]
        self.symbols = register_param('symbols', symbol_list)
        for sym in self.symbols: self.AddEquity(sym, Resolution.Minute)
            # note that the `AddEquity` resolution is `Minute`
            # this impacts how often `OnData` is called which determines whether
            # scheduled functions are called by Minute, Hour, or Daily
        
        # -----------------------------------------------------------------------------
        # init placeholders
        # -----------------------------------------------------------------------------
        
        self.openMarketOnOpenOrders = []

        self._longs = False
        self._shorts = False 
        
        # -----------------------------------------------------------------------------
        # other algo parameter settings
        # -----------------------------------------------------------------------------
        
        self.HOLDING_PERIOD = register_param('holding period (days)', 63)
        self.LOOKBACK = register_param('lookback (days)', 252)
        self.BET_SIZE = register_param('bet size', 0.05)
        self.LEVERAGE = register_param('leverage', 1.)

        # -----------------------------------------------------------------------------        
        # track RAM and computation time for main func, also leverage and cash 
        # ----------------------------------------------------------------------------- 
        self.splotName = 'Strategy Info'
        sPlot = Chart(self.splotName)
        sPlot.AddSeries(Series('RAM',  SeriesType.Line, 0))
        sPlot.AddSeries(Series('Time',  SeriesType.Line, 1))
        sPlot.AddSeries(Series('Leverage',  SeriesType.Line, 2))
        sPlot.AddSeries(Series('Cash',  SeriesType.Line, 3))        
        self.AddChart(sPlot)
        
        self.time_to_run_main_algo = 0    
        
        # -----------------------------------------------------------------------------
        # scheduled functions
        # -----------------------------------------------------------------------------
        # make buy list        
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday),
            self.TimeRules.AfterMarketOpen("SPY", 10),
            Action(self.run_main_algo))        
        
        # send orders
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday), 
            self.TimeRules.AfterMarketOpen("SPY", 30), 
            Action(self.send_orders))
        
        # check trade dates and liquidate if date condition
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday),
            self.TimeRules.AfterMarketOpen("SPY", 35),
            Action(self.check_liquidate))   
        
        # plot RAM
        self.Schedule.On(
            self.DateRules.EveryDay(), 
            self.TimeRules.AfterMarketOpen("SPY", 40),
            Action(self.CHART_RAM))
        
        # -----------------------------------------------------------------------------
        # initialize historical prices 
        #   cache the price data so we don't have to request the entire df for 
        #   every self.History() call
        # -----------------------------------------------------------------------------
        self.prices = (self.History(self.symbols, self.LOOKBACK, Resolution.Daily)
                       ["close"]
                       .unstack(level=0)
                       .astype(np.float32)) 
            
        # -----------------------------------------------------------------------------    
        # LOG PARAMETER REGISTRY
        #   this makes it easy to link backtest parameter settings with the saved results
        #   by logging/printing the information at the top of every backtest log
        # ----------------------------------------------------------------------------- 
        self.Debug('\n'+'-'*77+'\nPARAMETER REGISTRY\n{}...'.format(
            json.dumps(PARAMETER_REGISTRY, indent=2)
        ))
    
    def update_prices(self):
        """fn: to update prices in an efficient manner"""
        
        # get last date of stored prices
        most_recent_date = self.prices.index.max()
        current_date = self.Time
        # request only days that are missing from our dataset
        days_to_request = (current_date - most_recent_date).days
        
        # if prices up to date return 
        if days_to_request==0:
            return
        
        # get prices
        new_prices = (self.History(self.symbols, days_to_request, Resolution.Daily)
                      ["close"]
                      .unstack(level=0)
                      .astype(np.float32))
        self.prices = pd.concat([self.prices, new_prices]) # combine datasets
        # clean it up and keep only lookback period
        self.prices = self.prices.drop_duplicates().sort_index().iloc[-self.LOOKBACK:]
        return
    
    def check_liquidate(self):
        """fn: to check if todays date matches exit date and liquidate"""

        self.Log('\n'+'-'*77+'\n[{}] checking liquidation status...'.format(self.UtcTime))
                
        orders = self.Transactions.GetOrders(None)
        if orders: pass
        else: return
        
        # current time is gt_eq order time + holding period
        crit1 = lambda order: self.UtcTime >= (order.Time + timedelta(self.HOLDING_PERIOD))
        # order time is within today - holding period window
        #   7 day overlap between crit1 and crit2
        crit2 = lambda order: order.Time >= (self.UtcTime - timedelta(self.HOLDING_PERIOD + 7)) 
        
        for order in orders:
            if crit1(order) & crit2(order):
                if self.Portfolio[order.Symbol].Invested:
                    self.Liquidate(order.Symbol)
                    fmt_args = (self.UtcTime, order.Symbol, order.Time, self.UtcTime - order.Time)
                    self.Log('[{}] liquidating {}, order date: {}, time delta: {}'.format(*fmt_args))
        return
    
    def compute(self, sym):
        """fn: computation for bootstrapped confidence intervals for individual symbol"""
        
        train_px = self.prices[sym]
        train_df = make_returns(train_px)
        tmp_x = train_df.reshape(-1, 1)
        
        ### fit GMM ###
        gmm = make_gmm().fit(tmp_x)
        hidden_states = gmm.predict(tmp_x)
    
        ### get last state estimate ###
        last_state = hidden_states[-1]
        last_mean = gmm.means_[last_state]
        last_var = np.diag(gmm.covariances_[last_state])

        ### sample from distribution using last state parameters ###
        ### must match distribution selected in global parameter section ###
        
        ## normal distribution ##
        rvs = stats.norm.rvs(loc=last_mean, scale=np.sqrt(last_var), 
                             size=N_SAMPLES, random_state=RANDOM_STATE)
        low_ci, high_ci = stats.norm.interval(alpha=ALPHA,
                                              loc=np.mean(rvs), scale=np.std(rvs))
        
        ## laplace distribution ##
        #rvs = stats.laplace.rvs(loc=last_mean, scale=np.sqrt(last_var),
        #                        size=N_SAMPLES, random_state=RANDOM_STATE)
        #low_ci, high_ci = stats.laplace.interval(alpha=ALPHA,
        #                                         loc=np.mean(rvs), scale=np.std(rvs))
        
        ## johnson su distribution ##
        #rvs = stats.johnsonsu.rvs(a=a, b=b, 
        #                          loc=last_mean, scale=np.sqrt(last_var), 
        #                          size=N_SAMPLES, random_state=RANDOM_STATE)
        #low_ci, high_ci = stats.johnsonsu.interval(alpha=ALPHA,
        #                                           a=a, b=b, 
        #                                           loc=np.mean(rvs), scale=np.std(rvs))
        
        ## get current return ##
        tmp_ret = np.log(float(self.Securities[sym].Price) / train_px.iloc[-1])
        
        r_gt = (tmp_ret > high_ci)
        r_lt = (tmp_ret < low_ci)
        if r_gt: result_tag = 'too_high'
        elif r_lt: result_tag = 'too_low'
        else: result_tag = 'hit'
        
        ### row order: (symbol, low ci, high ci, current return, result_tag) ###
        sym_row = (sym, low_ci, high_ci, tmp_ret, result_tag)
        return sym_row
        
    def run_main_algo(self):
        """fn: run main algorithm computation"""
        
        start_time = time.time()
    
        self.Log('\n'+'-'*77+'\n[{}] Begin main algo computation...'.format(self.UtcTime))
        
        ### set buy/sell lists to False to confirm no carryover ###
        self._longs = False
        self._shorts = False
        
        ### update prices ###
        self.update_prices()
        
        ### compute data ###
        tmp_data_list = [self.compute(asset) 
                         for asset in self.prices.columns
                         if not self.Portfolio[asset].Invested]
        
        ### construct long/short arrays ###
        if tmp_data_list:
            cols = ['symbol', 'low_ci', 'high_ci', 'current_return', 'result_tag']
            df = (pd.DataFrame(tmp_data_list, columns=cols))
          
            self.Log('[{}] algo data:\n\t{}'.format(self.UtcTime, df)) 
            
            ### Choose between mean reversion algorithm ###
            self._longs = np.asarray(df.query('result_tag=="too_low"')['symbol'].unique())
            # self._shorts = np.asarray(df.query('result_tag=="too_high"')['symbol'].unique())            

            ### or breakout strategy ###
            # self._longs = np.asarray(df.query('result_tag=="too_high"')['symbol'].unique())            
            # self._shorts = np.asarray(df.query('result_tag=="too_low"')['symbol'].unique())

            log_str = (self.UtcTime, self._longs, self._shorts)
            self.Log('\n'+'-'*77+'\n[{0}] longs: {1}\n[{0}] shorts: {2}'.format(*log_str))
        else:
            self.Log('[{}] already fully invested, exiting...'.format(self.UtcTime))
            
        self.time_to_run_main_algo = time.time() - start_time
        return
    
    def send_orders(self):
        """fn: send orders"""
        
        self.Log('\n'+'-'*77+'\n[{}] checking L/S arrays to send orders...'.format(self.UtcTime))
        
        ### confirm lists are proper array datatype ###
        if isinstance(self._shorts, np.ndarray):
            if self._shorts.size: # confirm not empty 
                for sym in self._shorts:
                    if not self.Portfolio[sym].Invested: # only send order if not invested   
                        self.Log('[{}] sending short order for {}...'.format(self.UtcTime, sym))
                        short_shares = self.CalculateOrderQuantity(sym, -self.LEVERAGE*self.BET_SIZE)
                        newTicket = self.MarketOnOpenOrder(sym, short_shares)
                        self.openMarketOnOpenOrders.append(newTicket) # track ticket 
            else:
                self.Log('[{}] no shorts listed, no orders sent...'.format(self.UtcTime))
        
        ### confirm lists are proper array datatype ###        
        if isinstance(self._longs, np.ndarray):
            if self._longs.size: # confirm not empty 
                for sym in self._longs:
                    if not self.Portfolio[sym].Invested: # only send order if not invested 
                        self.Log('[{}] sending long order for {}...'.format(self.UtcTime, sym))
                        long_shares = self.CalculateOrderQuantity(sym, self.LEVERAGE*self.BET_SIZE)
                        newTicket = self.MarketOnOpenOrder(sym, long_shares)
                        self.openMarketOnOpenOrders.append(newTicket) # track ticket 
            else:
                self.Log('[{}] no longs listed, no orders sent...'.format(self.UtcTime))
                        
        return 

    def CHART_RAM(self):
        """fn: to track Ram, Computation Time, Leverage, Cash"""
        self.Plot(self.splotName,'RAM', OS.ApplicationMemoryUsed/1024.)
        self.Plot(self.splotName,'Time', self.time_to_run_main_algo)

        P = self.Portfolio
        self.track_leverage = P.TotalAbsoluteHoldingsCost / P.TotalPortfolioValue
        self.Plot(self.splotName, 'Leverage', float(self.track_leverage))
        self.Plot(self.splotName, 'Cash', float(self.Portfolio.Cash))   
        
    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm.
        Each new data point will be pumped in here.
        
        Not always necessary especially when using scheduled functions
        '''
        pass

     