# data_processing/noise_signal_treatment.py
"""
Noise and Signal Treatment Module

This module provides tools for filtering noise, smoothing data, and normalizing data.
"""
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

class DataSmoother:
    def __init__(self, config):
        self.ema_span = config.get('ema_span', 14)

    def smooth(self, series: pd.Series) -> pd.Series:
        """Applies EMA smoothing to a data series."""
        return series.ewm(span=self.ema_span, adjust=False).mean()

class VolumeNormalizer:
    def __init__(self, config):
        self.method = config.get('method', 'robust')

    def normalize(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Normalizes specified volume columns."""
        df_norm = df.copy()
        if self.method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        for col in columns:
            if col in df_norm.columns:
                df_norm[f'{col}_normalized'] = scaler.fit_transform(df_norm[[col]])
        return df_norm