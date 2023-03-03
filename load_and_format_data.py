import yfinance as yf
import pandas as pd 
import numpy as np

class Data:
    # Intialize inputs
    def __init__(self, tickers, start_date, end_date):
        self.tickers    = tickers
        self.start_date = start_date 
        self.end_date   = end_date 
        self.get_data()
       
    # Load the data and compute returns based on adjusted close (accounts for impact of dividends and splits). For time being, only return DataFrame with adjusted close and returns.
    def get_data(self):
        # Get data
        raw_data = yf.download(self.tickers, self.start_date, self.end_date)
        
        # Compute returns from adjusted close and return
        self.adj_close = raw_data.loc[:, 'Adj Close']
        self.rets      = self.adj_close.pct_change() 

    # Aggregate data to different frequency based on request.
    def aggregate_data(self, freq):
        # Aggregate using geometric compounding.
        self.agg_rets = (1 + self.rets).groupby(pd.Grouper(freq=freq)).prod() - 1
       


