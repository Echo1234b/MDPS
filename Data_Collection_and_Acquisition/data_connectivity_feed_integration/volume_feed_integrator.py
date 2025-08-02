# volume_feed_integrator.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import threading
import queue
import logging

class VolumeFeedIntegrator:
    """
    A class to collect tick-based or candle-based volume data from MT5.
    Normalizes volume metrics across different asset classes.
    """
    
    def __init__(self, mt5_connector, buffer_size=10000):
        """
        Initialize Volume Feed Integrator
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            buffer_size (int): Memory buffer size
        """
        self.mt5_connector = mt5_connector
        self.buffer_size = buffer_size
        self.volume_buffer = queue.Queue(maxsize=buffer_size)
        self.is_collecting = False
        self.collection_thread = None
        self.logger = self._setup_logger()
        self.volume_history = {}
        self.volume_multipliers = {}  # Multipliers to normalize volume across different assets
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("VolumeFeedIntegrator")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("volume_feed_integrator.log")
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
    
    def set_volume_multiplier(self, symbol, multiplier):
        """
        Set volume multiplier for a symbol to normalize volume across different assets
        
        Args:
            symbol (str): Trading symbol
            multiplier (float): Volume multiplier
        """
        self.volume_multipliers[symbol] = multiplier
        self.logger.info(f"Set volume multiplier for {symbol} to {multiplier}")
    
    def start_tick_volume_collection(self, symbol):
        """
        Start collecting tick-based volume data for specified symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: Whether successfully started collection
        """
        if self.is_collecting:
            self.logger.warning("Volume collection is already running")
            return False
        
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return False
        
        # Initialize volume history for symbol
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        
        # Get volume multiplier if not set
        if symbol not in self.volume_multipliers:
            # Default multiplier is 1.0
            self.volume_multipliers[symbol] = 1.0
            self.logger.info(f"Using default volume multiplier 1.0 for {symbol}")
        
        # Start collection thread
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collect_tick_volume, 
            args=(symbol,)
        )
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        self.logger.info(f"Started tick volume collection for {symbol}")
        return True
    
    def start_candle_volume_collection(self, symbol, timeframe):
        """
        Start collecting candle-based volume data for specified symbol and timeframe
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): MT5 timeframe constant
            
        Returns:
            bool: Whether successfully started collection
        """
        if self.is_collecting:
            self.logger.warning("Volume collection is already running")
            return False
        
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return False
        
        # Initialize volume history for symbol
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        
        # Get volume multiplier if not set
        if symbol not in self.volume_multipliers:
            # Default multiplier is 1.0
            self.volume_multipliers[symbol] = 1.0
            self.logger.info(f"Using default volume multiplier 1.0 for {symbol}")
        
        # Start collection thread
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collect_candle_volume, 
            args=(symbol, timeframe)
        )
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
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
        
        self.logger.info(f"Started candle volume collection for {symbol} ({timeframe_str})")
        return True
    
    def stop_collection(self):
        """
        Stop collecting volume data
        
        Returns:
            bool: Whether successfully stopped collection
        """
        if not self.is_collecting:
            self.logger.warning("Volume collection is not running")
            return False
        
        # Stop collection thread
        self.is_collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        self.logger.info("Stopped volume collection")
        return True
    
    def _collect_tick_volume(self, symbol):
        """
        Internal method to collect tick-based volume data, runs in separate thread
        
        Args:
            symbol (str): Trading symbol
        """
        last_time = None
        last_volume = 0
        
        while self.is_collecting:
            try:
                # Get current time
                current_time = datetime.now()
                
                # If first run or enough time has passed since last fetch, get new tick data
                if last_time is None or (current_time - last_time).total_seconds() >= 1:
                    # Get latest tick data
                    tick = mt5.symbol_info_tick(symbol)
                    
                    if tick is not None:
                        # Calculate volume change since last tick
                        volume_change = tick.volume - last_volume
                        last_volume = tick.volume
                        
                        # Apply volume multiplier
                        normalized_volume = volume_change * self.volume_multipliers[symbol]
                        
                        # Create volume data dictionary
                        volume_data = {
                            'time': tick.time,
                            'time_msc': tick.time_msc,
                            'symbol': symbol,
                            'volume': volume_change,
                            'normalized_volume': normalized_volume,
                            'type': 'tick'
                        }
                        
                        # Put volume data into buffer
                        if self.volume_buffer.full():
                            # If buffer is full, remove oldest data
                            try:
                                self.volume_buffer.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.volume_buffer.put(volume_data)
                        last_time = current_time
                        
                        # Update volume history
                        self.volume_history[symbol].append({
                            'time': tick.time,
                            'volume': normalized_volume
                        })
                        
                        # Keep only recent history (last 1000 entries)
                        if len(self.volume_history[symbol]) > 1000:
                            self.volume_history[symbol].pop(0)
                        
                        # Log
                        self.logger.debug(f"Collected tick volume for {symbol}: {volume_data}")
                    else:
                        self.logger.error(f"Failed to get tick for {symbol}")
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error collecting tick volume: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _collect_candle_volume(self, symbol, timeframe):
        """
        Internal method to collect candle-based volume data, runs in separate thread
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): MT5 timeframe constant
        """
        last_candle_time = None
        
        while self.is_collecting:
            try:
                # Get current time
                current_time = datetime.now()
                
                # If first run or enough time has passed since last fetch, get new candle data
                if last_candle_time is None or (current_time - last_candle_time).total_seconds() >= 60:
                    # Get latest candle data
                    candles = self.mt5_connector.get_last_n_rates(symbol, timeframe, 2)
                    
                    if candles is not None and len(candles) >= 2:
                        # Get the most recent completed candle (second to last)
                        candle = candles.iloc[-2]
                        candle_time = candle['time']
                        
                        # Check if this is a new candle
                        if last_candle_time is None or candle_time > last_candle_time:
                            last_candle_time = candle_time
                            
                            # Apply volume multiplier
                            normalized_volume = candle['tick_volume'] * self.volume_multipliers[symbol]
                            
                            # Create volume data dictionary
                            volume_data = {
                                'time': candle_time,
                                'symbol': symbol,
                                'volume': candle['tick_volume'],
                                'normalized_volume': normalized_volume,
                                'type': 'candle',
                                'timeframe': timeframe
                            }
                            
                            # Put volume data into buffer
                            if self.volume_buffer.full():
                                # If buffer is full, remove oldest data
                                try:
                                    self.volume_buffer.get_nowait()
                                except queue.Empty:
                                    pass
                            
                            self.volume_buffer.put(volume_data)
                            
                            # Update volume history
                            self.volume_history[symbol].append({
                                'time': candle_time,
                                'volume': normalized_volume
                            })
                            
                            # Keep only recent history (last 1000 entries)
                            if len(self.volume_history[symbol]) > 1000:
                                self.volume_history[symbol].pop(0)
                            
                            # Log
                            self.logger.debug(f"Collected candle volume for {symbol}: {volume_data}")
                    else:
                        self.logger.error(f"Failed to get candles for {symbol}")
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error collecting candle volume: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def get_volume_data(self, count=None):
        """
        Get volume data from buffer
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of volume data
        """
        data = []
        
        if count is None:
            # Get all data
            while not self.volume_buffer.empty():
                try:
                    data.append(self.volume_buffer.get_nowait())
                except queue.Empty:
                    break
        else:
            # Get specified number of records
            for _ in range(min(count, self.volume_buffer.qsize())):
                try:
                    data.append(self.volume_buffer.get_nowait())
                except queue.Empty:
                    break
        
        return data
    
    def get_volume_dataframe(self, count=None):
        """
        Get volume data from buffer and convert to DataFrame
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing volume data
        """
        data = self.get_volume_data(count)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        
        if 'time_msc' in df.columns:
            df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
        
        return df
    
    def get_volume_stats(self, symbol):
        """
        Get volume statistics for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Volume statistics
        """
        if symbol not in self.volume_history or not self.volume_history[symbol]:
            return {}
        
        volumes = [entry['volume'] for entry in self.volume_history[symbol]]
        
        return {
            'min': min(volumes),
            'max': max(volumes),
            'avg': sum(volumes) / len(volumes),
            'current': volumes[-1] if volumes else None,
            'samples': len(volumes)
        }
    
    def save_volume_to_csv(self, filename, count=None):
        """
        Save volume data in buffer to CSV file
        
        Args:
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_volume_dataframe(count)
            
            if df.empty:
                self.logger.warning("No volume data to save")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} volume records to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save volume data to CSV: {str(e)}")
            return False
    
    def save_volume_to_parquet(self, filename, count=None):
        """
        Save volume data in buffer to Parquet file
        
        Args:
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_volume_dataframe(count)
            
            if df.empty:
                self.logger.warning("No volume data to save")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} volume records to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save volume data to Parquet: {str(e)}")
            return False
