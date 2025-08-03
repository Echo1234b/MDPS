"""
Preprocessing & Feature Engineering Module
Generates technical indicators, features, and performs feature selection.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class FeatureEngine:
    """Main feature engineering class for financial data"""
    
    def __init__(self, config=None):
        self.config = config
        self.technical_indicator_generator = TechnicalIndicatorGenerator()
        self.feature_aggregator = FeatureAggregator()
        self.feature_selector = FeatureSelector()
        self.pattern_encoder = PatternEncoder()
        
    def generate_features(self, data):
        """Generate comprehensive feature set from market data"""
        logging.info("FeatureEngine: Starting feature generation")
        
        try:
            # Start with the input data
            features = data.copy()
            
            # 1. Generate technical indicators
            features = self.technical_indicator_generator.generate_indicators(features)
            
            # 2. Create lag features
            features = self._create_lag_features(features)
            
            # 3. Generate ratio and spread features
            features = self._create_ratio_features(features)
            
            # 4. Add time-based features
            features = self._add_time_features(features)
            
            # 5. Calculate rolling statistics
            features = self._calculate_rolling_stats(features)
            
            # 6. Generate pattern features
            features = self.pattern_encoder.encode_patterns(features)
            
            # 7. Aggregate multi-timeframe features (if configured)
            if self.config and hasattr(self.config, 'feature_settings'):
                features = self.feature_aggregator.aggregate_timeframes(features, self.config.feature_settings)
            
            # 8. Select top features (if configured)
            features = self.feature_selector.select_features(features)
            
            logging.info(f"FeatureEngine: Generated {len(features.columns)} features")
            return features
            
        except Exception as e:
            logging.error(f"FeatureEngine: Error in feature generation: {e}")
            raise

    def _create_lag_features(self, data, lags=[1, 2, 3, 5, 10]):
        """Create lagged features for price and volume data"""
        logging.info("FeatureEngine: Creating lag features")
        
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        lag_features = data.copy()
        
        for col in price_cols:
            if col in data.columns:
                for lag in lags:
                    lag_features[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return lag_features

    def _create_ratio_features(self, data):
        """Create ratio and spread features"""
        logging.info("FeatureEngine: Creating ratio features")
        
        ratio_features = data.copy()
        
        if all(col in data.columns for col in ['high', 'low', 'close']):
            # Price ratios
            ratio_features['hl_ratio'] = data['high'] / data['low']
            ratio_features['close_to_high'] = data['close'] / data['high']
            ratio_features['close_to_low'] = data['close'] / data['low']
            
        if all(col in data.columns for col in ['open', 'close']):
            ratio_features['open_close_ratio'] = data['open'] / data['close']
            ratio_features['body_size'] = abs(data['close'] - data['open']) / data['open']
            
        return ratio_features

    def _add_time_features(self, data):
        """Add time-based features"""
        logging.info("FeatureEngine: Adding time features")
        
        time_features = data.copy()
        
        # Extract time components if timestamp is available
        if hasattr(data.index, 'hour'):
            time_features['hour'] = data.index.hour
            time_features['day_of_week'] = data.index.dayofweek
            time_features['month'] = data.index.month
            
            # Cyclical encoding for time features
            time_features['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            time_features['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
            time_features['dow_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            time_features['dow_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
        
        return time_features

    def _calculate_rolling_stats(self, data, windows=[5, 10, 20, 50]):
        """Calculate rolling statistical features"""
        logging.info("FeatureEngine: Calculating rolling statistics")
        
        rolling_features = data.copy()
        price_cols = ['close', 'volume']
        
        for col in price_cols:
            if col in data.columns:
                for window in windows:
                    if len(data) >= window:
                        rolling_features[f'{col}_mean_{window}'] = data[col].rolling(window).mean()
                        rolling_features[f'{col}_std_{window}'] = data[col].rolling(window).std()
                        rolling_features[f'{col}_min_{window}'] = data[col].rolling(window).min()
                        rolling_features[f'{col}_max_{window}'] = data[col].rolling(window).max()
        
        return rolling_features

class TechnicalIndicatorGenerator:
    """Generates technical indicators for financial data"""
    
    def generate_indicators(self, data):
        """Generate comprehensive set of technical indicators"""
        logging.info("TechnicalIndicatorGenerator: Generating technical indicators")
        
        indicators = data.copy()
        
        if 'close' in data.columns:
            # Moving averages
            indicators = self._add_moving_averages(indicators)
            
            # Momentum indicators
            indicators = self._add_momentum_indicators(indicators)
            
            # Volatility indicators
            indicators = self._add_volatility_indicators(indicators)
            
            # Volume indicators
            indicators = self._add_volume_indicators(indicators)
        
        return indicators

    def _add_moving_averages(self, data, periods=[5, 10, 20, 50]):
        """Add moving average indicators"""
        for period in periods:
            if len(data) >= period:
                data[f'sma_{period}'] = data['close'].rolling(period).mean()
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        return data

    def _add_momentum_indicators(self, data):
        """Add momentum-based indicators"""
        # RSI
        if len(data) >= 14:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(data) >= 26:
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema12 - ema26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        return data

    def _add_volatility_indicators(self, data):
        """Add volatility-based indicators"""
        # Bollinger Bands
        if len(data) >= 20:
            sma20 = data['close'].rolling(20).mean()
            std20 = data['close'].rolling(20).std()
            data['bb_upper'] = sma20 + (std20 * 2)
            data['bb_lower'] = sma20 - (std20 * 2)
            data['bb_width'] = data['bb_upper'] - data['bb_lower']
            data['bb_position'] = (data['close'] - data['bb_lower']) / data['bb_width']
        
        # Average True Range
        if all(col in data.columns for col in ['high', 'low', 'close']):
            data['tr'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )
            if len(data) >= 14:
                data['atr'] = data['tr'].rolling(14).mean()
        
        return data

    def _add_volume_indicators(self, data):
        """Add volume-based indicators"""
        if 'volume' in data.columns:
            # Volume moving averages
            if len(data) >= 20:
                data['volume_sma_20'] = data['volume'].rolling(20).mean()
                data['volume_ratio'] = data['volume'] / data['volume_sma_20']
            
            # On-Balance Volume
            if 'close' in data.columns:
                price_change = data['close'].diff()
                obv = (price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * data['volume']).cumsum()
                data['obv'] = obv
        
        return data

class FeatureAggregator:
    """Aggregates features across different timeframes"""
    
    def aggregate_timeframes(self, data, feature_settings):
        """Aggregate features from multiple timeframes"""
        logging.info("FeatureAggregator: Aggregating multi-timeframe features")
        # Placeholder implementation
        return data

class FeatureSelector:
    """Selects most relevant features for modeling"""
    
    def select_features(self, data, max_features=50):
        """Select top features based on variance and correlation"""
        logging.info("FeatureSelector: Selecting relevant features")
        
        # Remove features with low variance
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        variance_threshold = 0.01
        
        selected_features = []
        for col in numeric_cols:
            if data[col].var() > variance_threshold:
                selected_features.append(col)
        
        # Limit number of features
        if len(selected_features) > max_features:
            selected_features = selected_features[:max_features]
        
        logging.info(f"FeatureSelector: Selected {len(selected_features)} features")
        return data[selected_features]

class PatternEncoder:
    """Encodes chart patterns and price action features"""
    
    def encode_patterns(self, data):
        """Encode basic price patterns"""
        logging.info("PatternEncoder: Encoding price patterns")
        
        patterns = data.copy()
        
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Doji pattern
            body_size = abs(data['close'] - data['open'])
            candle_range = data['high'] - data['low']
            patterns['is_doji'] = (body_size / candle_range < 0.1).astype(int)
            
            # Hammer pattern
            lower_shadow = data['open'].combine(data['close'], min) - data['low']
            upper_shadow = data['high'] - data['open'].combine(data['close'], max)
            patterns['is_hammer'] = ((lower_shadow > 2 * body_size) & 
                                   (upper_shadow < body_size)).astype(int)
            
            # Bullish/Bearish candles
            patterns['is_bullish'] = (data['close'] > data['open']).astype(int)
            patterns['is_bearish'] = (data['close'] < data['open']).astype(int)
        
        return patterns

__all__ = ['FeatureEngine']