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
import decimal as d
import json 
from algo_utils import *

# ------------------------------------------------------------------------------
# init parameter registry
# ------------------------------------------------------------------------------

PARAMETER_REGISTRY = {}

def register_param(name, value):
    PARAMETER_REGISTRY[name] = value
    return value

# -----------------------------------------------------------------------------
# algorithm class
# -----------------------------------------------------------------------------

class TradingWithGMM(QCAlgorithm):
    """
    """
    def Initialize(self):
        """Initial algorithm settings"""

        self.INIT_PORTFOLIO_CASH = register_param('portfolio starting cash', 100000)        

        self.SetStartDate(2008,9,1)  #Set Start Date
        self.SetEndDate(2019,12,31)    #Set End Date
        self.SetCash(self.INIT_PORTFOLIO_CASH) #Set Strategy Cash
        
        # -----------------------------------------------------------------------------
        # init brokerage model, important for realistic slippage/commission modeling
        # especially important if using leverage which requires margin account
        # -----------------------------------------------------------------------------
        
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage,
                               AccountType.Margin)

        # -----------------------------------------------------------------------------
        # init custom universe
        # -----------------------------------------------------------------------------
        
        self.BASE_SYMBOL = register_param('base symbol for algorithm management: ', 'SPY')   
        self.symbols = [self.BASE_SYMBOL, "QQQ", "DIA", "TLT", "GLD", "EFA", "EEM"]
        
        # QC updated symbols to use Symbol.ID instead of Symbol.Value
        # MUST CHANGE SYMBOLS IN OLDER ALGOS
        def ReIndex(ticker):
            e = self.AddEquity(ticker, Resolution.Minute)
            return f'{e.Symbol.ID}'
            
        self.symbols = [ReIndex(t) for t in self.symbols]
        # MUST UPDATE BASE SYMBOL TO NEW ID                       
        self.BASE_SYMBOL = [x for x in self.symbols if self.BASE_SYMBOL in x][0]        
        
        # -----------------------------------------------------------------------------
        # Algo Exchange Settings
        #   Using SPY here as it is synonymous with the Equity market. If trading diff
        #   assets like Futures then one would need to change the symbol used.
        # -----------------------------------------------------------------------------

        self.exchange = self.Securities[self.BASE_SYMBOL].Exchange

        # -----------------------------------------------------------------------------
        # other algo parameter settings
        # -----------------------------------------------------------------------------
        
        self.openMarketOnOpenOrders = []
        self._init_prices = False
        self._longs = False
        self._shorts = False 
        register_param('symbols: ', self.symbols)
        self._holding_period = register_param('holding period (days)', 30)
        self.LOOKBACK = register_param('historical lookback (days)', 252*3) # trading days        
        self.BET_SIZE = register_param('bet size (%)', 1/len(self.symbols)) #0.05)
    
        self.RANDOM_STATE = register_param('random_state', 777)
        self.ALPHA = register_param('gmm alpha', 0.95) # for sampling confidence intervals
        self.N_COMPONENTS = register_param('gmm n components', 2)        
        self.THRES = register_param('threshold probability for buy signal', 0.9)
        self.SAMPLES = register_param('number of samples for bootstrap', 1000)
        ## set resolution for historical data calls 
        self.HISTORY_RESOLUTION = Resolution.Daily 
        register_param('history api resolution', str(self.HISTORY_RESOLUTION)) 

        
        # -----------------------------------------------------------------------------
        # track RAM and computation time for main func, also leverage and cash
        # -----------------------------------------------------------------------------

        self.splotName = 'Strategy Info'
        sPlot = Chart(self.splotName)
        sPlot.AddSeries(Series('RAM',  SeriesType.Line, 0))
        sPlot.AddSeries(Series('Time',  SeriesType.Line, 1))
        sPlot.AddSeries(Series('Cash',  SeriesType.Line, 2))
        sPlot.AddSeries(Series('Leverage',  SeriesType.Line, 3))
        self.AddChart(sPlot)

        self.time_to_run_main_algo = 0

        # -----------------------------------------------------------------------------
        # scheduled functions
        # -----------------------------------------------------------------------------
        # make buy list  
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday),
            self.TimeRules.AfterMarketOpen(self.BASE_SYMBOL, 5),
            Action(self.init_prices))
            
        # make buy list  
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday),
            self.TimeRules.AfterMarketOpen(self.BASE_SYMBOL, 10),
            Action(self.run_main_algo))
            
        # send orders
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday),
            #self.DateRules.WeekStart(self.BASE_SYMBOL),
            #self.DateRules.EveryDay(self.BASE_SYMBOL),
            self.TimeRules.AfterMarketOpen(self.BASE_SYMBOL, 30),
            Action(self.send_orders))

        # check trade dates and liquidate if date condition
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Friday),
            #self.DateRules.WeekStart(self.BASE_SYMBOL),
            #self.DateRules.EveryDay(self.BASE_SYMBOL),
            self.TimeRules.AfterMarketOpen(self.BASE_SYMBOL, 35),
            Action(self.check_liquidate))

        # plot RAM
        self.Schedule.On(
            self.DateRules.EveryDay(self.BASE_SYMBOL),
            self.TimeRules.AfterMarketOpen(self.BASE_SYMBOL, 40),
            Action(self.CHART_RAM))


        # -----------------------------------------------------------------------------
        # LOG PARAMETER REGISTRY
        #   this makes it easy to link backtest parameter settings with the saved results
        #   by logging/printing the information at the top of every backtest log
        # -----------------------------------------------------------------------------
        self.Debug('\n'+'-'*77+'\nPARAMETER REGISTRY\n{}...'
                   .format(json.dumps(PARAMETER_REGISTRY, indent=2)))

    def init_prices(self):
        """
        fn: initialize historical prices
            cache the price data so we don't have to request the entire df every week
        """       
        if not self.symbols: self.Log('no symbols'); return
        #for sym in self.symbols: self.AddEquity(sym, Resolution.Minute)
        if self._init_prices: return 
        self.prices = (self.History(self.symbols,
                                    self.LOOKBACK, 
                                    self.HISTORY_RESOLUTION)
                       ["close"]
                       .unstack(level=0)
                       .astype(np.float32))
        self._init_prices=True
       
    def update_prices(self):
        """fn: to update prices in an efficient manner"""

        # get last date of stored prices
        most_recent_date = self.prices.index.max()
        current_date = self.Time
        # how many days do we need?
        days_to_request = how_many_days(current_date, most_recent_date)
        # if prices up to date return
        if zero_days_to_request(days_to_request): return
        # get new data
        new_prices = (self.History(self.symbols,
                                   days_to_request,
                                   self.HISTORY_RESOLUTION)
                      ["close"]
                      .unstack(level=0)
                      .astype(np.float32))
        # combine datasets
        self.prices = make_update_df(self.prices, new_prices, self.LOOKBACK)
        return
    
    def check_liquidate(self):
        """fn: to check if todays date matches exit date and liquidate
        """
        self.Log('\n'+'-'*77+'\n[{}] checking liquidation status...'.format(self.UtcTime))
                
        orders = self.Transactions.GetOrders(None)
        if orders: pass
        else: return
        
        # current time is gt_eq order time + holding period
        crit1 = lambda order: self.UtcTime >= (order.Time + timedelta(self._holding_period))
        # order time is within today - holding period window
        crit2 = lambda order: order.Time >= (self.UtcTime - timedelta(self._holding_period + 7)) # 7 day overlap
        
        for order in orders:
            if crit1(order) & crit2(order):
                if self.Portfolio[order.Symbol].Invested:
                    self.Liquidate(order.Symbol)
                    fmt_args = (self.UtcTime, order.Symbol, order.Time, self.UtcTime - order.Time)
                    self.Log('[{}] liquidating... {}, order date: {}, time delta: {}'.format(*fmt_args))
        return 
    
    def run_main_algo(self):
        """fn: run main algorithm"""
        self.Log('\n'+'-'*77+'\n[{}] Begin main algorithm computation...'.format(self.UtcTime))
        
        start_time = time.time() # timer
        self.update_prices() # update prices
        self._algo_data = False
        self._longs = list()
        self._shorts = list()

        for sym in self.prices.columns: # iterate through universe
            try:
                self.Log('checking symbol: {}'.format(str(sym)))
                pred_rows = list()
                # only compute if not already invested
                if (not self.Portfolio[sym].Invested):
                    
                    # symbol must be in list
                    train_px = self.prices.copy()
                    self.Debug('making training data for {}...'.format(str(sym)))
                    train_ts = make_returns(train_px)[sym].dropna()
                    train_ts = train_ts[np.isfinite(train_ts)]
                    if train_ts.empty:
                        self.Debug('{} train data is empty'.format(str(sym)))
                        continue
                    
                    if train_ts.shape[0] < 50:
                        self.Debug('{} train data has too few samples'.format(str(sym)))
                        continue

                    tmp_X_train = train_ts.values.reshape(-1, 1)
            
                    ### fit GMM ###
                    gmm = make_gmm(n_components=self.N_COMPONENTS, random_state=self.RANDOM_STATE).fit(tmp_X_train)
                    hidden_states = gmm.predict(tmp_X_train) # extract hidden states
                    hidden_state_prob = pd.DataFrame(gmm.predict_proba(tmp_X_train),
                                                     columns=['s1','s2'],
                                                     index=train_ts.index)
                    
                    state_df = train_ts.to_frame()
                    #self.Debug('state df:\n{}'.format(state_df.head()))
                    hs_prob_df = (pd.concat([state_df, hidden_state_prob],axis=1))
                    
                    # get state probability means and stds
                    s1_mu = hs_prob_df.query('abs(s1)>0.5')[sym].mean() 
                    s2_mu = hs_prob_df.query('abs(s2)>0.5')[sym].mean() 
                    
                    s1_std = hs_prob_df.query('abs(s1)>0.5')[sym].std() 
                    s2_std = hs_prob_df.query('abs(s2)>0.5')[sym].std()           
                    
                    ### get last state estimate ###
                    last_state = hidden_states[-1]
                    last_mean = gmm.means_[last_state][0]
                    last_var = np.diag(gmm.covariances_[last_state])[0]
                    
                    ### sample from distribution using last state parameters ###
                    #rvs = stats.norm.rvs(loc=last_mean, scale=np.sqrt(last_var), size=self.SAMPLES)
                    # sample directly from the fit
                    rvs = gmm.sample(self.SAMPLES)[0] 
                    low_ci, high_ci = stats.norm.interval(alpha=self.ALPHA, loc=np.mean(rvs), scale=np.std(rvs))
            
                    ## get current return ##
                    tmp_ret = np.log(float(self.Securities[sym].Price) / train_px[sym].iloc[-1])
                            
                    ### store data into rows ###
                    # columns: cols = ['Dates', 'ith_state', ith_ret','ith_std',
                    # 'low_ci', 'high_ci', 'actual_return', 'last_mean_class_0', 'last_mean_class_1',
                    # 'last_std_class_0', 'last_std_class_1', 'last_prob_class_0', 'last_prob_class_1',
                    # 'avg_class_0_mean', 'avg_class_1_mean', 'avg_class_0_std', 'avg_class_1_std']
                    
                    row = (train_ts.index[-1], last_state, last_mean, np.sqrt(last_var), 
                            low_ci, high_ci, tmp_ret,
                            gmm.means_.ravel()[0], gmm.means_.ravel()[1],
                            np.sqrt(np.diag(gmm.covariances_[0]))[0],
                             np.sqrt(np.diag(gmm.covariances_[1]))[0],
                            hidden_state_prob.iloc[-1][0], hidden_state_prob.iloc[-1][1],
                            s1_mu,s2_mu,s1_std,s2_std)                    
                    pred_rows.append(row) 
                    self.Debug('{} rowzz:\n{}'.format(str(sym), row))
                    
                if pred_rows:
                    cols = ['Dates', 'ith_state', 'ith_ret','ith_std',
                            'low_ci', 'high_ci', 'current_return',
                            'last_mean_class_0', 'last_mean_class_1',
                            'last_std_class_0', 'last_std_class_1',
                            'last_prob_class_0', 'last_prob_class_1',
                            'avg_class_0_mean', 'avg_class_1_mean',
                            'avg_class_0_std', 'avg_class_1_std']             

                    pred_df = make_final_pred_df(pred_rows, cols, self.THRES, sym)
                    if pred_df.iloc[-1].loc['buys']==1: self._longs.append(sym)

                    # self._longs = np.asarray(df.query('result_tag=="too_high"')['symbol'].unique())
                    # self._shorts = np.asarray(df.query('result_tag=="too_low"')['symbol'].unique())
                    
                    #self.Log('\n'+'-'*77+'\n[{0}] longs: {1}\n[{0}] shorts: {2}'.format(self.UtcTime, self._longs, self._shorts))
                else:
                    self.Debug('missing or invested in {}'.format(sym))
                    
            except Exception as e:
                self.Debug('{} error: {}'.format(sym, e))
                continue

        ## end timer
        self.time_to_run_main_algo = time.time() - start_time
        self.Plot(self.splotName, 'Time', self.time_to_run_main_algo)                
        return 
    
    def send_orders(self):
        """fn: send orders"""
        
        self.Log('\n'+'-'*77+'\n[{}] checking buy sell arrays to send orders...'.format(self.UtcTime))

        if self._longs:
            for sym in self._longs:
                if not self.Portfolio[sym].Invested:
                    self.Log('[{}] sending long order for {}...'.format(self.UtcTime, sym))                        
                    newTicket = self.MarketOnOpenOrder(sym, self.CalculateOrderQuantity(sym, self.BET_SIZE))
                    self.openMarketOnOpenOrders.append(newTicket)
        else:
            self.Log('send_orders >> no longs listed, no orders sent...')            
        return
    
    
    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm.
        Each new data point will be pumped in here.'''

        pass

    def CHART_RAM(self):
        # Once a day or something reasonable to prevent spam
        self.Plot(self.splotName,'RAM', OS.ApplicationMemoryUsed/1024.)
        P = self.Portfolio
        self.track_account_leverage = P.TotalAbsoluteHoldingsCost/P.TotalPortfolioValue
        self.Plot(self.splotName, 'Leverage', float(self.track_account_leverage))
        self.Plot(self.splotName, 'Cash', float(self.Portfolio.Cash))
        return
