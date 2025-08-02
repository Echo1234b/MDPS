# bid_ask_streamer.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import threading
import queue
import logging

class BidAskStreamer:
    """
    A class to stream real-time bid/ask spreads from MT5.
    Calculates and logs spread metrics, detects spread anomalies.
    """
    
    def __init__(self, mt5_connector, buffer_size=10000):
        """
        Initialize Bid/Ask streamer
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            buffer_size (int): Memory buffer size
        """
        self.mt5_connector = mt5_connector
        self.buffer_size = buffer_size
        self.bid_ask_buffer = queue.Queue(maxsize=buffer_size)
        self.is_streaming = False
        self.streaming_thread = None
        self.logger = self._setup_logger()
        self.spread_history = []
        self.spread_threshold = None
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("BidAskStreamer")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("bid_ask_streamer.log")
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
    
    def set_spread_threshold(self, threshold):
        """
        Set spread threshold for anomaly detection
        
        Args:
            threshold (float): Spread threshold in points
        """
        self.spread_threshold = threshold
        self.logger.info(f"Set spread threshold to {threshold} points")
    
    def start_streaming(self, symbol):
        """
        Start streaming bid/ask data for specified symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: Whether successfully started streaming
        """
        if self.is_streaming:
            self.logger.warning("Bid/Ask streaming is already running")
            return False
        
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return False
        
        # Start streaming thread
        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._stream_bid_ask, args=(symbol,))
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
        self.logger.info(f"Started bid/ask streaming for {symbol}")
        return True
    
    def stop_streaming(self):
        """
        Stop streaming bid/ask data
        
        Returns:
            bool: Whether successfully stopped streaming
        """
        if not self.is_streaming:
            self.logger.warning("Bid/Ask streaming is not running")
            return False
        
        # Stop streaming thread
        self.is_streaming = False
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=5)
        
        self.logger.info("Stopped bid/ask streaming")
        return True
    
    def _stream_bid_ask(self, symbol):
        """
        Internal method to stream bid/ask data, runs in separate thread
        
        Args:
            symbol (str): Trading symbol
        """
        last_time = None
        
        while self.is_streaming:
            try:
                # Get current time
                current_time = datetime.now()
                
                # If first run or enough time has passed since last fetch, get new tick data
                if last_time is None or (current_time - last_time).total_seconds() >= 0.5:
                    # Get latest tick data
                    tick = mt5.symbol_info_tick(symbol)
                    
                    if tick is not None:
                        # Calculate spread in points
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info is not None:
                            point = symbol_info.point
                            spread_points = (tick.ask - tick.bid) / point
                            
                            # Create bid/ask data dictionary
                            bid_ask_data = {
                                'time': tick.time,
                                'time_msc': tick.time_msc,
                                'bid': tick.bid,
                                'ask': tick.ask,
                                'last': tick.last,
                                'spread_points': spread_points,
                                'volume': tick.volume,
                                'flags': tick.flags,
                                'volume_real': tick.volume_real
                            }
                            
                            # Put bid/ask data into buffer
                            if self.bid_ask_buffer.full():
                                # If buffer is full, remove oldest data
                                try:
                                    self.bid_ask_buffer.get_nowait()
                                except queue.Empty:
                                    pass
                            
                            self.bid_ask_buffer.put(bid_ask_data)
                            last_time = current_time
                            
                            # Update spread history
                            self.spread_history.append(spread_points)
                            if len(self.spread_history) > 1000:  # Keep only recent history
                                self.spread_history.pop(0)
                            
                            # Check for spread anomalies if threshold is set
                            if self.spread_threshold is not None and spread_points > self.spread_threshold:
                                self.logger.warning(f"Spread anomaly detected for {symbol}: {spread_points} points")
                            
                            # Log
                            self.logger.debug(f"Streamed bid/ask for {symbol}: {bid_ask_data}")
                        else:
                            self.logger.error(f"Failed to get symbol info for {symbol}")
                    else:
                        self.logger.error(f"Failed to get tick for {symbol}")
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error streaming bid/ask: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def get_bid_ask_data(self, count=None):
        """
        Get bid/ask data from buffer
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of bid/ask data
        """
        data = []
        
        if count is None:
            # Get all data
            while not self.bid_ask_buffer.empty():
                try:
                    data.append(self.bid_ask_buffer.get_nowait())
                except queue.Empty:
                    break
        else:
            # Get specified number of records
            for _ in range(min(count, self.bid_ask_buffer.qsize())):
                try:
                    data.append(self.bid_ask_buffer.get_nowait())
                except queue.Empty:
                    break
        
        return data
    
    def get_bid_ask_dataframe(self, count=None):
        """
        Get bid/ask data from buffer and convert to DataFrame
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing bid/ask data
        """
        data = self.get_bid_ask_data(count)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        
        if 'time_msc' in df.columns:
            df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
        
        return df
    
    def get_spread_stats(self):
        """
        Get spread statistics
        
        Returns:
            dict: Spread statistics
        """
        if not self.spread_history:
            return {}
        
        return {
            'min': min(self.spread_history),
            'max': max(self.spread_history),
            'avg': sum(self.spread_history) / len(self.spread_history),
            'current': self.spread_history[-1] if self.spread_history else None,
            'samples': len(self.spread_history)
        }
    
    def save_bid_ask_to_csv(self, filename, count=None):
        """
        Save bid/ask data in buffer to CSV file
        
        Args:
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_bid_ask_dataframe(count)
            
            if df.empty:
                self.logger.warning("No bid/ask data to save")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} bid/ask records to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save bid/ask data to CSV: {str(e)}")
            return False
    
    def save_bid_ask_to_parquet(self, filename, count=None):
        """
        Save bid/ask data in buffer to Parquet file
        
        Args:
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_bid_ask_dataframe(count)
            
            if df.empty:
                self.logger.warning("No bid/ask data to save")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} bid/ask records to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save bid/ask data to Parquet: {str(e)}")
            return False
