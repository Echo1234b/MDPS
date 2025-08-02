# ohlcv_extractor.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import os

class OHLCVExtractor:
    """
    A class to extract structured OHLCV data for any instrument and timeframe.
    Handles missing bars, corporate actions, and data adjustments.
    """
    
    def __init__(self, mt5_connector, cache_dir="cache/ohlcv_data"):
        """
        Initialize OHLCV Extractor
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            cache_dir (str): Directory to cache extracted data
        """
        self.mt5_connector = mt5_connector
        self.cache_dir = cache_dir
        self.logger = self._setup_logger()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("OHLCVExtractor")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("ohlcv_extractor.log")
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_cache_filename(self, symbol, timeframe, start_date, end_date):
        """
        Generate cache filename for OHLCV data
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): MT5 timeframe constant
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            str: Cache filename
        """
        # Convert timeframe to string
        timeframe_str = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1",
            mt5.TIMEFRAME_W1: "W1",
            mt5.TIMEFRAME_MN1: "MN1"
        }.get(timeframe, "UNKNOWN")
        
        # Format dates as strings
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        # Generate filename
        filename = f"{symbol}_{timeframe_str}_{start_str}_{end_str}.parquet"
        return os.path.join(self.cache_dir, filename)
    
    def _load_from_cache(self, filename):
        """
        Load OHLCV data from cache
        
        Args:
            filename (str): Cache filename
            
        Returns:
            pandas.DataFrame: Cached data or None if not found
        """
        if not os.path.exists(filename):
            return None
        
        try:
            df = pd.read_parquet(filename)
            self.logger.info(f"Loaded OHLCV data from cache: {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load OHLCV data from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, df, filename):
        """
        Save OHLCV data to cache
        
        Args:
            df (pandas.DataFrame): Data to cache
            filename (str): Cache filename
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df.to_parquet(filename)
            self.logger.info(f"Saved OHLCV data to cache: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save OHLCV data to cache: {str(e)}")
            return False
    
    def _detect_missing_bars(self, df, timeframe):
        """
        Detect missing bars in OHLCV data
        
        Args:
            df (pandas.DataFrame): OHLCV data
            timeframe (int): MT5 timeframe constant
            
        Returns:
            list: List of missing datetime ranges
        """
        if df.empty or 'time' not in df.columns:
            return []
        
        # Sort by time
        df = df.sort_values('time')
        
        # Determine expected time interval based on timeframe
        interval_minutes = {
            mt5.TIMEFRAME_M1: 1,
            mt5.TIMEFRAME_M5: 5,
            mt5.TIMEFRAME_M15: 15,
            mt5.TIMEFRAME_M30: 30,
            mt5.TIMEFRAME_H1: 60,
            mt5.TIMEFRAME_H4: 240,
            mt5.TIMEFRAME_D1: 1440,
            mt5.TIMEFRAME_W1: 10080,
            mt5.TIMEFRAME_MN1: 43200
        }.get(timeframe, 60)
        
        interval = timedelta(minutes=interval_minutes)
        
        # Check for missing bars
        missing_ranges = []
        expected_time = df.iloc[0]['time']
        
        for _, row in df.iterrows():
            actual_time = row['time']
            
            # While expected time is less than actual time, there are missing bars
            while expected_time < actual_time:
                missing_ranges.append((expected_time, expected_time + interval))
                expected_time += interval
            
            # Move to next expected time
            expected_time = actual_time + interval
        
        return missing_ranges
    
    def _fill_missing_bars(self, df, timeframe, missing_ranges):
        """
        Fill missing bars in OHLCV data
        
        Args:
            df (pandas.DataFrame): OHLCV data
            timeframe (int): MT5 timeframe constant
            missing_ranges (list): List of missing datetime ranges
            
        Returns:
            pandas.DataFrame: OHLCV data with filled bars
        """
        if df.empty or not missing_ranges:
            return df
        
        # Create DataFrame for missing bars
        missing_data = []
        
        for start_time, end_time in missing_ranges:
            # Find the previous and next bars
            prev_bar = df[df['time'] < start_time].iloc[-1] if len(df[df['time'] < start_time]) > 0 else None
            next_bar = df[df['time'] >= end_time].iloc[0] if len(df[df['time'] >= end_time]) > 0 else None
            
            # Determine OHLCV values for missing bar
            if prev_bar is not None and next_bar is not None:
                # Interpolate between previous and next bars
                open_price = prev_bar['close']
                close_price = next_bar['open']
                high_price = max(open_price, close_price)
                low_price = min(open_price, close_price)
                
                # Use average volume
                volume = (prev_bar['tick_volume'] + next_bar['tick_volume']) / 2
            elif prev_bar is not None:
                # Use previous bar values
                open_price = prev_bar['close']
                close_price = prev_bar['close']
                high_price = prev_bar['close']
                low_price = prev_bar['close']
                volume = prev_bar['tick_volume']
            elif next_bar is not None:
                # Use next bar values
                open_price = next_bar['open']
                close_price = next_bar['open']
                high_price = next_bar['open']
                low_price = next_bar['open']
                volume = next_bar['tick_volume']
            else:
                # Use default values
                open_price = 0
                close_price = 0
                high_price = 0
                low_price = 0
                volume = 0
            
            # Add missing bar
            missing_data.append({
                'time': start_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'tick_volume': volume,
                'real_volume': 0,
                'spread': 0,
                'is_missing': True
            })
        
        # Create DataFrame for missing bars
        missing_df = pd.DataFrame(missing_data)
        
        # Combine with original data
        combined_df = pd.concat([df, missing_df], ignore_index=True)
        
        # Sort by time
        combined_df = combined_df.sort_values('time')
        
        # Reset index
        combined_df = combined_df.reset_index(drop=True)
        
        return combined_df
    
    def extract_ohlcv(self, symbol, timeframe, start_date, end_date, fill_missing=True, use_cache=True):
        """
        Extract OHLCV data for specified symbol, timeframe, and date range
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): MT5 timeframe constant
            start_date (datetime): Start date
            end_date (datetime): End date
            fill_missing (bool): Whether to fill missing bars
            use_cache (bool): Whether to use cache
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return None
        
        # Check cache first
        cache_filename = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        if use_cache:
            cached_data = self._load_from_cache(cache_filename)
            if cached_data is not None:
                return cached_data
        
        # Get OHLCV data
        self.logger.info(f"Extracting OHLCV data for {symbol} from {start_date} to {end_date}")
        df = self.mt5_connector.get_rates(symbol, timeframe, start_date, end_date)
        
        if df is None or df.empty:
            self.logger.error(f"No OHLCV data returned for {symbol} from {start_date} to {end_date}")
            return None
        
        # Add column to identify missing bars
        df['is_missing'] = False
        
        # Detect and fill missing bars if requested
        if fill_missing:
            missing_ranges = self._detect_missing_bars(df, timeframe)
            if missing_ranges:
                self.logger.info(f"Detected {len(missing_ranges)} missing bars for {symbol}")
                df = self._fill_missing_bars(df, timeframe, missing_ranges)
        
        # Save to cache
        if use_cache:
            self._save_to_cache(df, cache_filename)
        
        return df
    
    def extract_recent_ohlcv(self, symbol, timeframe, count, fill_missing=True, use_cache=True):
        """
        Extract recent OHLCV data for specified symbol and timeframe
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): MT5 timeframe constant
            count (int): Number of bars to extract
            fill_missing (bool): Whether to fill missing bars
            use_cache (bool): Whether to use cache
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return None
        
        # Get recent OHLCV data
        self.logger.info(f"Extracting recent OHLCV data for {symbol}, count: {count}")
        df = self.mt5_connector.get_last_n_rates(symbol, timeframe, count)
        
        if df is None or df.empty:
            self.logger.error(f"No OHLCV data returned for {symbol}, count: {count}")
            return None
        
        # Add column to identify missing bars
        df['is_missing'] = False
        
        # Detect and fill missing bars if requested
        if fill_missing:
            missing_ranges = self._detect_missing_bars(df, timeframe)
            if missing_ranges:
                self.logger.info(f"Detected {len(missing_ranges)} missing bars for {symbol}")
                df = self._fill_missing_bars(df, timeframe, missing_ranges)
        
        # Generate cache filename based on current time
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Arbitrary range for filename
        cache_filename = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        
        # Save to cache
        if use_cache:
            self._save_to_cache(df, cache_filename)
        
        return df
    
    def adjust_for_dividends(self, df, symbol, dividends):
        """
        Adjust OHLCV data for dividends
        
        Args:
            df (pandas.DataFrame): OHLCV data
            symbol (str): Trading symbol
            dividends (list): List of dividend events, each with 'date' and 'amount'
            
        Returns:
            pandas.DataFrame: Adjusted OHLCV data
        """
        if df.empty or not dividends:
            return df
        
        # Make a copy of the DataFrame
        adjusted_df = df.copy()
        
        # Sort dividends by date
        dividends = sorted(dividends, key=lambda x: x['date'])
        
        # For each dividend, adjust prices before the ex-dividend date
        for dividend in dividends:
            ex_date = dividend['date']
            amount = dividend['amount']
            
            # Find the index of the bar before the ex-dividend date
            mask = adjusted_df['time'] < ex_date
            if not mask.any():
                continue
            
            # Get the last bar before the ex-dividend date
            last_idx = adjusted_df[mask].index[-1]
            
            # Calculate adjustment factor
            close_price = adjusted_df.loc[last_idx, 'close']
            if close_price > 0:
                adjustment_factor = 1 - (amount / close_price)
                
                # Adjust all prices before the ex-dividend date
                adjusted_df.loc[:last_idx, 'open'] *= adjustment_factor
                adjusted_df.loc[:last_idx, 'high'] *= adjustment_factor
                adjusted_df.loc[:last_idx, 'low'] *= adjustment_factor
                adjusted_df.loc[:last_idx, 'close'] *= adjustment_factor
                
                self.logger.info(f"Adjusted prices for {symbol} before {ex_date} for dividend of {amount}")
        
        return adjusted_df
    
    def adjust_for_splits(self, df, symbol, splits):
        """
        Adjust OHLCV data for stock splits
        
        Args:
            df (pandas.DataFrame): OHLCV data
            symbol (str): Trading symbol
            splits (list): List of split events, each with 'date' and 'ratio'
            
        Returns:
            pandas.DataFrame: Adjusted OHLCV data
        """
        if df.empty or not splits:
            return df
        
        # Make a copy of the DataFrame
        adjusted_df = df.copy()
        
        # Sort splits by date
        splits = sorted(splits, key=lambda x: x['date'])
        
        # For each split, adjust prices before the split date
        for split in splits:
            split_date = split['date']
            ratio = split['ratio']
            
            # Find the index of the bar before the split date
            mask = adjusted_df['time'] < split_date
            if not mask.any():
                continue
            
            # Get the last bar before the split date
            last_idx = adjusted_df[mask].index[-1]
            
            # Adjust all prices before the split date
            adjusted_df.loc[:last_idx, 'open'] *= ratio
            adjusted_df.loc[:last_idx, 'high'] *= ratio
            adjusted_df.loc[:last_idx, 'low'] *= ratio
            adjusted_df.loc[:last_idx, 'close'] *= ratio
            
            # Adjust volume (inverse of price adjustment)
            adjusted_df.loc[:last_idx, 'tick_volume'] /= ratio
            adjusted_df.loc[:last_idx, 'real_volume'] /= ratio
            
            self.logger.info(f"Adjusted prices and volume for {symbol} before {split_date} for split ratio of {ratio}")
        
        return adjusted_df
    
    def clear_cache(self, older_than_days=None):
        """
        Clear cache
        
        Args:
            older_than_days (int): Remove files older than this many days, None means remove all
            
        Returns:
            int: Number of files removed
        """
        removed_count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                
                # Check if it's a file
                if os.path.isfile(filepath):
                    # If older_than_days is specified, check file age
                    if older_than_days is not None:
                        file_time = os.path.getmtime(filepath)
                        file_age_days = (time.time() - file_time) / (24 * 3600)
                        
                        if file_age_days <= older_than_days:
                            continue
                    
                    # Remove file
                    os.remove(filepath)
                    removed_count += 1
                    self.logger.info(f"Removed cache file: {filepath}")
            
            self.logger.info(f"Cleared {removed_count} cache files")
            return removed_count
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
            return 0
