"""
Data Cleaning & Signal Processing Module
Handles data quality assurance, temporal alignment, noise filtering, and signal processing.
"""

import logging
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import zscore

class DataCleaner:
    """Main data cleaning and signal processing class"""
    
    def __init__(self, config=None):
        self.config = config
        self.missing_value_handler = MissingValueHandler()
        self.outlier_detector = OutlierDetector()
        self.noise_filter = NoiseFilter()
        self.data_smoother = DataSmoother()
        self.timestamp_normalizer = TimestampNormalizer()
        
    def process(self, data):
        """Main processing pipeline for data cleaning and signal processing"""
        logging.info("DataCleaner: Starting data cleaning and signal processing")
        
        try:
            # 1. Handle missing values
            data = self.missing_value_handler.handle_missing(data)
            
            # 2. Normalize timestamps
            data = self.timestamp_normalizer.normalize(data)
            
            # 3. Remove duplicates
            data = self._remove_duplicates(data)
            
            # 4. Detect and handle outliers
            data = self.outlier_detector.detect_and_handle(data)
            
            # 5. Apply noise filtering
            data = self.noise_filter.filter_noise(data)
            
            # 6. Smooth data if configured
            if self.config and getattr(self.config, 'cleaning_settings', {}).get('smooth_window'):
                data = self.data_smoother.smooth(data, self.config.cleaning_settings['smooth_window'])
            
            logging.info("DataCleaner: Data cleaning completed successfully")
            return data
            
        except Exception as e:
            logging.error(f"DataCleaner: Error in processing: {e}")
            raise

    def _remove_duplicates(self, data):
        """Remove duplicate entries based on timestamp"""
        if 'timestamp' in data.columns:
            initial_count = len(data)
            data = data.drop_duplicates(subset=['timestamp'], keep='last')
            removed_count = initial_count - len(data)
            if removed_count > 0:
                logging.info(f"DataCleaner: Removed {removed_count} duplicate entries")
        return data

class MissingValueHandler:
    """Handles missing values in time series data"""
    
    def handle_missing(self, data):
        """Handle missing values using various strategies"""
        logging.info("MissingValueHandler: Processing missing values")
        
        # Check for missing values
        missing_info = data.isnull().sum()
        if missing_info.sum() > 0:
            logging.info(f"MissingValueHandler: Found missing values: {missing_info.to_dict()}")
            
            # Forward fill for price data
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(method='ffill')
            
            # Fill volume with 0 or median
            if 'volume' in data.columns:
                data['volume'] = data['volume'].fillna(0)
            
            # Drop rows that still have NaN values
            data = data.dropna()
            
        return data

class OutlierDetector:
    """Detects and handles outliers in financial data"""
    
    def detect_and_handle(self, data, z_threshold=3):
        """Detect outliers using z-score method"""
        logging.info("OutlierDetector: Detecting outliers")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series(False, index=data.index)
        
        for col in numeric_cols:
            if col not in ['timestamp']:  # Skip timestamp column
                z_scores = np.abs(zscore(data[col], nan_policy='omit'))
                col_outliers = z_scores > z_threshold
                outlier_mask |= col_outliers
                
                outlier_count = col_outliers.sum()
                if outlier_count > 0:
                    logging.info(f"OutlierDetector: Found {outlier_count} outliers in {col}")
        
        # Log total outliers but don't remove them for financial data
        total_outliers = outlier_mask.sum()
        if total_outliers > 0:
            logging.info(f"OutlierDetector: Total outliers detected: {total_outliers} (keeping for analysis)")
        
        return data

class NoiseFilter:
    """Filters noise from price and volume data"""
    
    def filter_noise(self, data):
        """Apply noise filtering to price data"""
        logging.info("NoiseFilter: Applying noise filtering")
        
        filtered_data = data.copy()
        
        # Apply Savitzky-Golay filter to price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in filtered_data.columns and len(filtered_data) > 5:
                try:
                    filtered_data[f'{col}_filtered'] = signal.savgol_filter(
                        filtered_data[col], 
                        window_length=min(5, len(filtered_data) // 2 * 2 + 1), 
                        polyorder=2
                    )
                except Exception as e:
                    logging.warning(f"NoiseFilter: Could not filter {col}: {e}")
        
        return filtered_data

class DataSmoother:
    """Smooths data using various methods"""
    
    def smooth(self, data, window_size=5):
        """Apply smoothing to data"""
        logging.info(f"DataSmoother: Applying smoothing with window size {window_size}")
        
        smoothed_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['timestamp'] and len(data) >= window_size:
                smoothed_data[f'{col}_smooth'] = data[col].rolling(
                    window=window_size, 
                    center=True
                ).mean()
        
        return smoothed_data

class TimestampNormalizer:
    """Normalizes and validates timestamps"""
    
    def normalize(self, data):
        """Normalize timestamp format and timezone"""
        logging.info("TimestampNormalizer: Normalizing timestamps")
        
        if 'timestamp' in data.columns:
            # Ensure timestamp is datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Set timestamp as index if not already
            if data.index.name != 'timestamp':
                data = data.set_index('timestamp')
        
        return data

__all__ = ['DataCleaner']
