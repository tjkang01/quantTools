import yfinance as yf
import pandas as pd 
import numpy as np
import heapq as hq

class Data:
    # Intialize inputs
    def __init__(self, tickers, start_date, end_date, min_sample=0.05):
        self.tickers       = tickers
        self.start_date    = start_date 
        self.end_date      = end_date 
        self.min_sample    = min_sample
        self.ann_factor    = 252
        self.get_data()
       
    # Load the data and compute returns based on adjusted close (accounts for impact of dividends and splits). For time being, only return DataFrame with adjusted close and returns.
    def get_data(self):
        # Get data
        raw_data = yf.download(tickers=self.tickers, start=self.start_date, end=self.end_date)
        
        # Compute returns from adjusted close and return
        self.adj_close = raw_data.loc[:, ['Adj Close']]
        self.rets      = self.adj_close.pct_change().fillna(0).rename(columns={'Adj Close': 'Returns'})

    # Aggregate data to different frequency based on request.
    def aggregate_data(self, freq):
        # Aggregate using geometric compounding.
        self.agg_rets = (1 + self.rets).groupby(pd.Grouper(freq=freq)).prod() - 1

    # For a given set of returns, estimate returns normalized by rolling exponentially-weighted volatility, for a given half-life. Optionally, user can provide a specific volatility to normalize to. Allows user to more easily compare assets with different vols.
    def normalize_rets(self, rets, half_life, new_vol=1):
        # Define alpha factor
        alpha = 1 - 0.5**(1/half_life)

        # Define the minimum number of points as 5% of the sample
        min_periods = round(self.min_sample * rets.shape[0])

        # Compute volatility and shift by one day 
        vol = rets.ewm(alpha=alpha, min_periods=min_periods).std().shift(1) * np.sqrt(self.ann_factor)

        # Return the scaled estimates
        self.norm_rets = rets / vol * new_vol


# Compute simple summary statistics over a specified time range for some set of returns.
def compute_summary(rets, ann_factor=252, start_date=None, end_date=None):
    # Get datetime index and set default values to start and end if none are provided.
    index     = rets.index 
    if start_date is None:
        start_date = index[0]
    if end_date is None:
        end_date   = index[-1]

    # Find nearest indexes corresponding to these dates - this deals with case where user provides an invalid date. 
    start_idx = index.get_indexer([start_date], method='nearest')[0]
    end_idx   = index.get_indexer([end_date], method='nearest')[0] + 1

    # Obtain the relevant data slice
    rets_sliced = rets.iloc[start_idx:end_idx, :]

    # Compute the number of days with NaN values
    num_nan = (~rets_sliced.isna()).sum()

    # Compute the annualized returns, annualized vol, and ratio
    ann_ret = ((1 + rets_sliced).prod())**(ann_factor / num_nan) - 1
    ann_vol = rets_sliced.std() * np.sqrt(ann_factor) 
    sharpe_ratio = ann_ret / ann_vol

    # Compile results into one DataFrame
    summary_stats = pd.concat([ann_ret, ann_vol, sharpe_ratio], axis=1)
    summary_stats.columns = ['AnnRet', 'AnnVol', 'SharpeRatio']
    return summary_stats

'''
Compute maximum drawdown for a single return series: the maximum drawdown is the worst
loss you would have experienced over the whole lifetime of an investment (assuming 
you held it at a constant weight the entire time). Returns the value of the maximum 
drawdown, the index of the peak, the index of the trough, and the index of the 
recovery. If the drawdown does not recover before the end of the time series, 
the index is set to NaN. 
'''
def compute_max_drawdown(rets):
    # First, convert the returns to a cumulative price series. 
    prices = (1 + rets).cumprod()

    # Compute the cumulative maximum and possible drawdowns 
    cum_max  = prices.cummax()
    drawdown = prices / cum_max - 1

    # Find the minimum value from drawdown as the worst drawdown.
    trough_idx = np.argmin(drawdown)
    max_dd     = drawdown.iloc[trough_idx, 0]

    # To find the index of the peak value, find the first instance of the cumulative maximum corresponding to the one at the trough. 
    peak_idx = np.argmax(cum_max.iloc[:trough_idx])

    # Finally to find the recovery index, look at the cumulative maximum after the trough index. If the cumulative maximum ever exceeds that of the value at the trough, then the drawdown is recovered.
    max_at_trough = cum_max.iloc[trough_idx]
    has_recovered = cum_max.iloc[trough_idx+1:] > max_at_trough
    recovery_idx  = prices.shape[0] - 1
    if has_recovered.any()[0]:
        recovery_idx = np.argmax(has_recovered) + trough_idx + 1
    
    # Return values as a list for mutability
    return [max_dd, peak_idx, trough_idx, recovery_idx]

'''
Returns the worst n drawdowns, if possible. min_points controls whether or 
not a segment of data gets added to the queue of considered returns series. 
The core logic works by computing finding a maximum drawdown and bisecting the
time series based on when the drawdown starts and ends. To deal with the fact
that losses may be concentrated on one particular side of the time series, we
use a min-heap to keep track of and pop off the worst drawdown at each point.
min_points controls whether or not we consider evaluating a segment of the 
time series; trivially short segments may not be informative.
'''
def compute_n_drawdowns(rets, num_dd=5, min_points=20):
    # Setup the dataframe to store values
    dd_info = pd.DataFrame(columns=['Drawdown', 'Peak Point', 'Trough Point', 'Recovery Point'])
    dd_size = 0

    # The min-heap that keeps track of the worst drawdowns we have seen so far. 
    # It prioritizes based on the drawdown value.
    dd_heap = []

    # Define a queue for each portion of the time series we evaluate. The queue holds 
    # tuples of (return series, offset). As the drawdown function returns indexes corresponding
    # to the various points of a drawdown, the offset allows us to keep track of where
    # each return series is relative to the bisection point.
    returns_queue = [(rets, 0)]    

    # Get the underlying index used in the original data frame for consistency
    rets_index = rets.index

    # Evaluate until either the desired number of drawdowns is found or there
    # are no more segments left to consider
    while dd_size < num_dd and returns_queue:
        # Evaluate drawdowns from the queue
        while returns_queue:
            # Pop from the front and compute drawdowns
            portion_returns, offset = returns_queue.pop(0) 
            drawdown_stats          = compute_max_drawdown(portion_returns)

            # Keep track of the two segments that flank the drawdown
            segments     = []
            # Add left segment if it there are enough points
            left_portion = portion_returns.iloc[:drawdown_stats[1]]
            if left_portion.shape[0] >= min_points:
                segments.append((left_portion, offset))

            # Same with the right segment. 
            right_portion = portion_returns.iloc[drawdown_stats[-1]:]
            if  right_portion.shape[0] >= min_points:
                segments.append((right_portion, drawdown_stats[-1] + offset))

            # Move the indexes accordingly based on the offset
            drawdown_stats[1: ] = list(map(lambda x: x + offset, drawdown_stats[1: ]))

            # Convert to the appropriate index
            drawdown_stats[1: ] = list(map(lambda x: rets_index[x], drawdown_stats[1: ]))

            # Push the drawdown statistics to the heap along with the segments.
            to_push = (drawdown_stats, segments)
            hq.heappush(dd_heap, to_push)
        
        # Push the worst drawdown onto the final dataframe.
        to_record            = hq.heappop(dd_heap)
        dd_info.loc[dd_size] = to_record[0]
        dd_size += 1

        # Record the segments flanking the popped drawdown into the queue.
        for i in range(len(to_record[1])):
            returns_queue.append(to_record[1][i])
    
    # Finally, if any drawdowns have a recovery index that corresponds to the 
    # last date in the time series, the drawdown has not recovered yet. We update accordingly
    any_ongoing = (dd_info.loc[:, 'Recovery Point'] == rets_index[-1])
    dd_info.loc[any_ongoing, 'Recovery Point'] = 'Ongoing'
    return dd_info