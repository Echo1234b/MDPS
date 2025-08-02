# historical_data_loader.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import os
import pytz

class HistoricalDataLoader:
    """
    A class to download OHLCV data of any timeframe and granularity.
    Handles date range selection, symbol validation, and data chunking.
    """
    
    def __init__(self, mt5_connector, cache_dir="cache/historical_data"):
        """
        Initialize Historical Data Loader
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            cache_dir (str): Directory to cache downloaded data
        """
        self.mt5_connector = mt5_connector
        self.cache_dir = cache_dir
        self.logger = self._setup_logger()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("HistoricalDataLoader")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("historical_data_loader.log")
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
        Generate cache filename for historical data
        
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
        Load historical data from cache
        
        Args:
            filename (str): Cache filename
            
        Returns:
            pandas.DataFrame: Cached data or None if not found
        """
        if not os.path.exists(filename):
            return None
        
        try:
            df = pd.read_parquet(filename)
            self.logger.info(f"Loaded historical data from cache: {filename}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load historical data from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, df, filename):
        """
        Save historical data to cache
        
        Args:
            df (pandas.DataFrame): Data to cache
            filename (str): Cache filename
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df.to_parquet(filename)
            self.logger.info(f"Saved historical data to cache: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save historical data to cache: {str(e)}")
            return False
    
    def load_historical_data(self, symbol, timeframe, start_date, end_date, use_cache=True, chunk_size=None):
        """
        Load historical OHLCV data
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): MT5 timeframe constant
            start_date (datetime): Start date
            end_date (datetime): End date
            use_cache (bool): Whether to use cache
            chunk_size (int): Number of days to fetch at once, None means all at once
            
        Returns:
            pandas.DataFrame: Historical OHLCV data
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
        
        # Convert dates to UTC if they're timezone-aware
        if start_date.tzinfo is not None:
            start_date = start_date.astimezone(pytz.UTC).replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.astimezone(pytz.UTC).replace(tzinfo=None)
        
        # Initialize result DataFrame
        result_df = pd.DataFrame()
        
        # Determine chunk size if not specified
        if chunk_size is None:
            # Default chunk size based on timeframe
            if timeframe in [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5]:
                chunk_size = 30  # 30 days
            elif timeframe in [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30]:
                chunk_size = 90  # 90 days
            elif timeframe in [mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]:
                chunk_size = 365  # 1 year
            else:
                chunk_size = 3650  # 10 years
        
        # Fetch data in chunks if date range is large
        if (end_date - start_date).days > chunk_size:
            current_start = start_date
            
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=chunk_size), end_date)
                
                self.logger.info(f"Fetching data for {symbol} from {current_start} to {current_end}")
                
                # Get data for current chunk
                chunk_df = self.mt5_connector.get_rates(symbol, timeframe, current_start, current_end)
                
                if chunk_df is not None and not chunk_df.empty:
                    # Append to result
                    result_df = pd.concat([result_df, chunk_df], ignore_index=True)
                else:
                    self.logger.warning(f"No data returned for {symbol} from {current_start} to {current_end}")
                
                # Move to next chunk
                current_start = current_end
                
                # Brief pause to avoid overwhelming the server
                time.sleep(0.5)
        else:
            # Fetch all data at once
            self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            result_df = self.mt5_connector.get_rates(symbol, timeframe, start_date, end_date)
        
        # Sort by time
        if not result_df.empty:
            result_df = result_df.sort_values('time')
            
            # Save to cache
            if use_cache:
                self._save_to_cache(result_df, cache_filename)
            
            return result_df
        else:
            self.logger.error(f"No data returned for {symbol} from {start_date} to {end_date}")
            return None
    
    def load_recent_data(self, symbol, timeframe, count, use_cache=True):
        """
        Load recent OHLCV data
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): MT5 timeframe constant
            count (int): Number of bars to load
            use_cache (bool): Whether to use cache
            
        Returns:
            pandas.DataFrame: Recent OHLCV data
        """
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return None
        
        # Get recent data
        self.logger.info(f"Fetching recent data for {symbol}, count: {count}")
        df = self.mt5_connector.get_last_n_rates(symbol, timeframe, count)
        
        if df is not None and not df.empty:
            # Sort by time
            df = df.sort_values('time')
            
            # Generate cache filename based on current time
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Arbitrary range for filename
            cache_filename = self._get_cache_filename(symbol, timeframe, start_date, end_date)
            
            # Save to cache
            if use_cache:
                self._save_to_cache(df, cache_filename)
            
            return df
        else:
            self.logger.error(f"No data returned for {symbol}, count: {count}")
            return None
    
    def load_tick_data(self, symbol, start_date, end_date, use_cache=True):
        """
        Load historical tick data
        
        Args:
            symbol (str): Trading symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            use_cache (bool): Whether to use cache
            
        Returns:
            pandas.DataFrame: Historical tick data
        """
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return None
        
        # Check cache first
        cache_filename = self._get_cache_filename(symbol, "TICKS", start_date, end_date)
        if use_cache:
            cached_data = self._load_from_cache(cache_filename)
            if cached_data is not None:
                return cached_data
        
        # Convert dates to UTC if they're timezone-aware
        if start_date.tzinfo is not None:
            start_date = start_date.astimezone(pytz.UTC).replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.astimezone(pytz.UTC).replace(tzinfo=None)
        
        # Fetch tick data
        self.logger.info(f"Fetching tick data for {symbol} from {start_date} to {end_date}")
        df = self.mt5_connector.get_ticks(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            # Sort by time
            df = df.sort_values('time')
            
            # Save to cache
            if use_cache:
                self._save_to_cache(df, cache_filename)
            
            return df
        else:
            self.logger.error(f"No tick data returned for {symbol} from {start_date} to {end_date}")
            return None
    
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
