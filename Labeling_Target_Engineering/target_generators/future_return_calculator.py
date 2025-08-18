import pandas as pd
import numpy as np
import talib

class FutureReturnCalculator:
    def __init__(self, price_data, return_windows=[1, 3, 5]):
        self.price_data = price_data
        self.return_windows = return_windows

    def calculate_returns(self, threshold=0.0):
        returns = {}
        for window in self.return_windows:
            future_returns = self.price_data['close'].pct_change(window).shift(-window)
            returns[f'return_{window}'] = future_returns
            
            # Apply threshold-based labeling
            returns[f'label_{window}'] = np.where(future_returns > threshold, 1,
                                                 np.where(future_returns < -threshold, -1, 0))
        return pd.DataFrame(returns)

    def calculate_log_returns(self):
        log_returns = {}
        for window in self.return_windows:
            future_log_returns = np.log(self.price_data['close'] / self.price_data['close'].shift(window)).shift(-window)
            log_returns[f'log_return_{window}'] = future_log_returns
        return pd.DataFrame(log_returns)
