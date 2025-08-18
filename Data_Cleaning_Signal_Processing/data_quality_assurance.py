# data_processing/data_quality_assurance.py
"""
Data Quality Assurance Module

This module provides tools for handling missing values, removing duplicates,
detecting outliers, and sanitizing financial data.
"""
import pandas as pd
import numpy as np
from scipy import stats

class MissingValueHandler:
    def __init__(self, config):
        self.strategy = config.get('strategy', 'ffill')
        
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values based on the configured strategy."""
        if self.strategy == 'ffill':
            return df.fillna(method='ffill')
        elif self.strategy == 'bfill':
            return df.fillna(method='bfill')
        elif self.strategy == 'mean':
            return df.fillna(df.mean())
        elif self.strategy == 'drop':
            return df.dropna()
        return df

class DuplicateEntryRemover:
    def remove(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate rows from the DataFrame."""
        return df.drop_duplicates()

class OutlierDetector:
    def __init__(self, config):
        self.method = config.get('method', 'z_score')
        self.threshold = config.get('threshold', 3.0)
        self.handling_method = config.get('handling_method', 'cap')

    def handle(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Detects and handles outliers in specified columns."""
        for col in columns:
            if self.method == 'z_score' and col in df.columns:
                z_scores = stats.zscore(df[col].dropna())
                outlier_mask = np.abs(z_scores) > self.threshold
                
                if self.handling_method == 'remove':
                    df = df[~outlier_mask]
                elif self.handling_method == 'cap':
                    cap_value = df[col].std() * self.threshold
                    df[col] = df[col].clip(lower=df[col].mean() - cap_value, upper=df[col].mean() + cap_value)
        return df

class DataSanitizer:
    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs basic data sanitization like type conversion and rounding."""
        df_sanitized = df.copy()
        # تحويل أنواع البيانات لتحسين استهلاك الذاكرة
        for col in df_sanitized.select_dtypes(include=[np.number]):
            if df_sanitized[col].dtype == 'float64':
                df_sanitized[col] = df_sanitized[col].astype(np.float32)
            if df_sanitized[col].dtype == 'int64':
                df_sanitized[col] = df_sanitized[col].astype(np.int32)
        return df_sanitized