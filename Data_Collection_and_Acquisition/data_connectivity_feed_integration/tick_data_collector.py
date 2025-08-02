# tick_data_collector.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import queue
import logging

class TickDataCollector:
    """
    A class to continuously collect tick-level data (bid/ask/last volume) from MT5.
    Timestamps and stores for high-resolution analysis and feature engineering.
    """
    
    def __init__(self, mt5_connector, buffer_size=10000):
        """
        Initialize Tick data collector
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            buffer_size (int): Memory buffer size
        """
        self.mt5_connector = mt5_connector
        self.buffer_size = buffer_size
        self.tick_buffer = queue.Queue(maxsize=buffer_size)
        self.is_collecting = False
        self.collection_thread = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("TickDataCollector")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("tick_data_collector.log")
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
    
    def start_collection(self, symbol):
        """
        Start collecting tick data for specified symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: Whether successfully started collection
        """
        if self.is_collecting:
            self.logger.warning("Tick collection is already running")
            return False
        
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return False
        
        # Start collection thread
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collect_ticks, args=(symbol,))
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        self.logger.info(f"Started tick collection for {symbol}")
        return True
    
    def stop_collection(self):
        """
        Stop collecting tick data
        
        Returns:
            bool: Whether successfully stopped collection
        """
        if not self.is_collecting:
            self.logger.warning("Tick collection is not running")
            return False
        
        # Stop collection thread
        self.is_collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        self.logger.info("Stopped tick collection")
        return True
    
    def _collect_ticks(self, symbol):
        """
        Internal method to collect tick data, runs in separate thread
        
        Args:
            symbol (str): Trading symbol
        """
        last_time = None
        
        while self.is_collecting:
            try:
                # Get current time
                current_time = datetime.now()
                
                # If first run or enough time has passed since last fetch, get new tick data
                if last_time is None or (current_time - last_time).total_seconds() >= 1:
                    # Get latest tick data
                    ticks = mt5.symbol_info_tick(symbol)
                    
                    if ticks is not None:
                        # Create tick data dictionary
                        tick_data = {
                            'time': ticks.time,
                            'bid': ticks.bid,
                            'ask': ticks.ask,
                            'last': ticks.last,
                            'volume': ticks.volume,
                            'time_msc': ticks.time_msc,
                            'flags': ticks.flags,
                            'volume_real': ticks.volume_real
                        }
                        
                        # Put tick data into buffer
                        if self.tick_buffer.full():
                            # If buffer is full, remove oldest data
                            try:
                                self.tick_buffer.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.tick_buffer.put(tick_data)
                        last_time = current_time
                        
                        # Log
                        self.logger.debug(f"Collected tick for {symbol}: {tick_data}")
                    else:
                        self.logger.error(f"Failed to get tick for {symbol}")
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error collecting ticks: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def get_ticks(self, count=None):
        """
        Get tick data from buffer
        
        Args:
            count (int): Number of ticks to get, None means get all
            
        Returns:
            list: List of tick data
        """
        ticks = []
        
        if count is None:
            # Get all tick data
            while not self.tick_buffer.empty():
                try:
                    ticks.append(self.tick_buffer.get_nowait())
                except queue.Empty:
                    break
        else:
            # Get specified number of tick data
            for _ in range(min(count, self.tick_buffer.qsize())):
                try:
                    ticks.append(self.tick_buffer.get_nowait())
                except queue.Empty:
                    break
        
        return ticks
    
    def get_ticks_dataframe(self, count=None):
        """
        Get tick data from buffer and convert to DataFrame
        
        Args:
            count (int): Number of ticks to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing tick data
        """
        ticks = self.get_ticks(count)
        
        if not ticks:
            return pd.DataFrame()
        
        df = pd.DataFrame(ticks)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        
        if 'time_msc' in df.columns:
            df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
        
        return df
    
    def save_ticks_to_csv(self, filename, count=None):
        """
        Save tick data in buffer to CSV file
        
        Args:
            filename (str): Filename
            count (int): Number of ticks to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_ticks_dataframe(count)
            
            if df.empty:
                self.logger.warning("No tick data to save")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} ticks to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save ticks to CSV: {str(e)}")
            return False
    
    def save_ticks_to_parquet(self, filename, count=None):
        """
        Save tick data in buffer to Parquet file
        
        Args:
            filename (str): Filename
            count (int): Number of ticks to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_ticks_dataframe(count)
            
            if df.empty:
                self.logger.warning("No tick data to save")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} ticks to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save ticks to Parquet: {str(e)}")
            return False
