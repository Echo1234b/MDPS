# data_processing/temporal_structural_alignment.py
"""
Temporal and Structural Alignment Module

This module provides tools for normalizing timestamps and converting data frequencies.
"""
import pandas as pd

class TimestampNormalizer:
    def __init__(self, config):
        self.timestamp_col = config.get('timestamp_col', 'time')
        self.timezone = config.get('timezone', 'UTC')

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the timestamp column to a datetime index with a specific timezone."""
        df_norm = df.copy()
        if self.timestamp_col in df_norm.columns:
            df_norm[self.timestamp_col] = pd.to_datetime(df_norm[self.timestamp_col], unit='s')
            df_norm = df_norm.set_index(self.timestamp_col)
            if df_norm.index.tz is None:
                df_norm = df_norm.index.tz_localize('UTC').tz_convert(self.timezone)
        return df_norm

class DataFrequencyConverter:
    def __init__(self, config):
        self.freq = config.get('freq', '1H')

    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resamples the DataFrame to a new target frequency."""
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        # استخدم الأعمدة الموجودة فقط
        valid_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
        return df.resample(self.freq).agg(valid_rules).dropna()