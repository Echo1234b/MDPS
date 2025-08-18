import pandas as pd
import numpy as np
from typing import Dict, List

class MultiScaleFeatures:
    def __init__(self):
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    def merge_timeframe_features(self, data: Dict[str, pd.DataFrame], target_tf: str = '1m') -> pd.DataFrame:
        target_data = data[target_tf].copy()
        
        for tf, df in data.items():
            if tf != target_tf:
                ratio = int(tf[:-1]) // int(target_tf[:-1])
                aligned_data = df.resample(f'{target_tf}').ffill()
                for col in df.columns:
                    target_data[f'{col}_{tf}'] = aligned_data[col]
        
        return target_data
    
    def create_lag_features(self, data: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        df = data.copy()
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
    
    def calculate_rolling_stats(self, data: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        df = data.copy()
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
        return df
