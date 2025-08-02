# adaptive_sampling_controller.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging

class AdaptiveSamplingController:
    """
    A class to dynamically adjust sampling frequency based on market volatility and tick frequency.
    Optimizes data collection efficiency while preserving critical information.
    """
    
    def __init__(self, mt5_connector, min_interval=0.1, max_interval=60.0, volatility_threshold=0.002):
        """
        Initialize Adaptive Sampling Controller
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            min_interval (float): Minimum sampling interval in seconds
            max_interval (float): Maximum sampling interval in seconds
            volatility_threshold (float): Volatility threshold for interval adjustment
        """
        self.mt5_connector = mt5_connector
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.volatility_threshold = volatility_threshold
        self.logger = self._setup_logger()
        
        # Sampling parameters
        self.current_interval = (min_interval + max_interval) / 2
        self.target_symbols = set()
        self.symbol_intervals = {}
        self.symbol_volatilities = {}
        self.symbol_tick_counts = {}
        self.last_sample_times = {}
        self.symbol_prices = {}  # Store last price for each symbol
        self.symbol_price_history = {}  # Store price history for volatility calculation
        
        # Control status
        self.is_controlling = False
        self.control_thread = None
        
        # Callbacks
        self.sample_callbacks = {}
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("AdaptiveSamplingController")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("adaptive_sampling_controller.log")
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
    
    def add_symbol(self, symbol, callback=None):
        """
        Add symbol to adaptive sampling control
        
        Args:
            symbol (str): Trading symbol
            callback (function): Callback function to handle sampled data
            
        Returns:
            bool: Whether successfully added symbol
        """
        if symbol in self.target_symbols:
            self.logger.warning(f"Symbol {symbol} is already being sampled")
            return False
        
        # Add to target symbols
        self.target_symbols.add(symbol)
        
        # Initialize symbol parameters
        self.symbol_intervals[symbol] = self.current_interval
        self.symbol_volatilities[symbol] = 0.0
        self.symbol_tick_counts[symbol] = 0
        self.last_sample_times[symbol] = time.time()
        self.symbol_prices[symbol] = None
        self.symbol_price_history[symbol] = []
        
        # Set callback if provided
        if callback is not None:
            self.sample_callbacks[symbol] = callback
        
        # If this is the first symbol, start the control thread
        if not self.is_controlling:
            self.is_controlling = True
            self.control_thread = threading.Thread(target=self._control_sampling)
            self.control_thread.daemon = True
            self.control_thread.start()
        
        self.logger.info(f"Added symbol {symbol} to adaptive sampling control")
        return True
    
    def remove_symbol(self, symbol):
        """
        Remove symbol from adaptive sampling control
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: Whether successfully removed symbol
        """
        if symbol not in self.target_symbols:
            self.logger.warning(f"Symbol {symbol} is not being sampled")
            return False
        
        # Remove from target symbols
        self.target_symbols.remove(symbol)
        
        # Remove symbol parameters
        del self.symbol_intervals[symbol]
        del self.symbol_volatilities[symbol]
        del self.symbol_tick_counts[symbol]
        del self.last_sample_times[symbol]
        del self.symbol_prices[symbol]
        del self.symbol_price_history[symbol]
        
        # Remove callback if exists
        if symbol in self.sample_callbacks:
            del self.sample_callbacks[symbol]
        
        # If no more symbols, stop the control thread
        if not self.target_symbols and self.is_controlling:
            self.is_controlling = False
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=5)
        
        self.logger.info(f"Removed symbol {symbol} from adaptive sampling control")
        return True
    
    def set_sample_callback(self, symbol, callback):
        """
        Set sample callback for a symbol
        
        Args:
            symbol (str): Trading symbol
            callback (function): Callback function to handle sampled data
            
        Returns:
            bool: Whether successfully set callback
        """
        if symbol not in self.target_symbols:
            self.logger.warning(f"Symbol {symbol} is not being sampled")
            return False
        
        self.sample_callbacks[symbol] = callback
        self.logger.info(f"Set sample callback for symbol {symbol}")
        return True
    
    def set_interval_limits(self, min_interval, max_interval):
        """
        Set interval limits
        
        Args:
            min_interval (float): Minimum sampling interval in seconds
            max_interval (float): Maximum sampling interval in seconds
            
        Returns:
            bool: Whether successfully set limits
        """
        if min_interval >= max_interval:
            self.logger.error("Minimum interval must be less than maximum interval")
            return False
        
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.logger.info(f"Set interval limits: min={min_interval}s, max={max_interval}s")
        return True
    
    def set_volatility_threshold(self, threshold):
        """
        Set volatility threshold for interval adjustment
        
        Args:
            threshold (float): Volatility threshold
            
        Returns:
            bool: Whether successfully set threshold
        """
        if threshold <= 0:
            self.logger.error("Volatility threshold must be positive")
            return False
        
        self.volatility_threshold = threshold
        self.logger.info(f"Set volatility threshold to {threshold}")
        return True
    
    def start_control(self):
        """
        Start adaptive sampling control
        
        Returns:
            bool: Whether successfully started control
        """
        if self.is_controlling:
            self.logger.warning("Adaptive sampling control is already running")
            return False
        
        if not self.target_symbols:
            self.logger.error("No symbols to control")
            return False
        
        # Start control thread
        self.is_controlling = True
        self.control_thread = threading.Thread(target=self._control_sampling)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        self.logger.info("Started adaptive sampling control")
        return True
    
    def stop_control(self):
        """
        Stop adaptive sampling control
        
        Returns:
            bool: Whether successfully stopped control
        """
        if not self.is_controlling:
            self.logger.warning("Adaptive sampling control is not running")
            return False
        
        # Stop control thread
        self.is_controlling = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=5)
        
        self.logger.info("Stopped adaptive sampling control")
        return True
    
    def _control_sampling(self):
        """
        Internal method to control sampling, runs in separate thread
        """
        while self.is_controlling:
            try:
                # Process each symbol
                for symbol in self.target_symbols:
                    # Get current time
                    current_time = time.time()
                    
                    # Check if it's time to sample
                    if current_time - self.last_sample_times[symbol] >= self.symbol_intervals[symbol]:
                        # Get tick data
                        tick = self.mt5_connector.get_symbol_info_tick(symbol)
                        
                        if tick is not None:
                            # Update tick count
                            self.symbol_tick_counts[symbol] += 1
                            
                            # Get price
                            price = (tick.bid + tick.ask) / 2
                            
                            # Update price history
                            self.symbol_price_history[symbol].append({
                                'time': current_time,
                                'price': price
                            })
                            
                            # Keep only recent history (last 100 ticks)
                            if len(self.symbol_price_history[symbol]) > 100:
                                self.symbol_price_history[symbol].pop(0)
                            
                            # Calculate volatility if we have enough data
                            if len(self.symbol_price_history[symbol]) >= 10:
                                # Get prices from history
                                prices = [entry['price'] for entry in self.symbol_price_history[symbol]]
                                
                                # Calculate volatility as standard deviation of returns
                                returns = []
                                for i in range(1, len(prices)):
                                    if prices[i-1] != 0:
                                        returns.append((prices[i] - prices[i-1]) / prices[i-1])
                                
                                if returns:
                                    volatility = np.std(returns)
                                    self.symbol_volatilities[symbol] = volatility
                                    
                                    # Adjust sampling interval based on volatility
                                    if volatility > self.volatility_threshold:
                                        # High volatility, decrease interval (sample more frequently)
                                        new_interval = max(
                                            self.min_interval,
                                            self.symbol_intervals[symbol] * 0.9
                                        )
                                    else:
                                        # Low volatility, increase interval (sample less frequently)
                                        new_interval = min(
                                            self.max_interval,
                                            self.symbol_intervals[symbol] * 1.1
                                        )
                                    
                                    # Update interval if changed significantly
                                    if abs(new_interval - self.symbol_intervals[symbol]) > 0.1:
                                        self.symbol_intervals[symbol] = new_interval
                                        self.logger.debug(f"Adjusted sampling interval for {symbol} to {new_interval:.2f}s (volatility: {volatility:.4f})")
                            
                            # Update last price
                            self.symbol_prices[symbol] = price
                            
                            # Update last sample time
                            self.last_sample_times[symbol] = current_time
                            
                            # Create sample data
                            sample_data = {
                                'time': datetime.fromtimestamp(current_time),
                                'symbol': symbol,
                                'bid': tick.bid,
                                'ask': tick.ask,
                                'last': tick.last,
                                'volume': tick.volume,
                                'sampling_interval': self.symbol_intervals[symbol],
                                'volatility': self.symbol_volatilities[symbol],
                                'tick_count': self.symbol_tick_counts[symbol]
                            }
                            
                            # Call callback if provided
                            if symbol in self.sample_callbacks:
                                try:
                                    self.sample_callbacks[symbol](sample_data)
                                except Exception as e:
                                    self.logger.error(f"Error in sample callback for {symbol}: {str(e)}")
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in adaptive sampling control: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def get_sampling_interval(self, symbol):
        """
        Get current sampling interval for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Sampling interval in seconds or None if symbol not found
        """
        if symbol not in self.target_symbols:
            self.logger.warning(f"Symbol {symbol} is not being sampled")
            return None
        
        return self.symbol_intervals[symbol]
    
    def get_volatility(self, symbol):
        """
        Get current volatility for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Volatility or None if symbol not found
        """
        if symbol not in self.target_symbols:
            self.logger.warning(f"Symbol {symbol} is not being sampled")
            return None
        
        return self.symbol_volatilities[symbol]
    
    def get_tick_count(self, symbol):
        """
        Get tick count for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            int: Tick count or None if symbol not found
        """
        if symbol not in self.target_symbols:
            self.logger.warning(f"Symbol {symbol} is not being sampled")
            return None
        
        return self.symbol_tick_counts[symbol]
    
    def get_price_history(self, symbol, count=None):
        """
        Get price history for a symbol
        
        Args:
            symbol (str): Trading symbol
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of price records or None if symbol not found
        """
        if symbol not in self.target_symbols:
            self.logger.warning(f"Symbol {symbol} is not being sampled")
            return None
        
        if count is None:
            return self.symbol_price_history[symbol].copy()
        else:
            return self.symbol_price_history[symbol][-count:]
    
    def get_price_history_dataframe(self, symbol, count=None):
        """
        Get price history for a symbol as DataFrame
        
        Args:
            symbol (str): Trading symbol
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing price history or None if symbol not found
        """
        history = self.get_price_history(symbol, count)
        
        if history is None:
            return None
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def reset_symbol_stats(self, symbol):
        """
        Reset statistics for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: Whether successfully reset
        """
        if symbol not in self.target_symbols:
            self.logger.warning(f"Symbol {symbol} is not being sampled")
            return False
        
        # Reset statistics
        self.symbol_volatilities[symbol] = 0.0
        self.symbol_tick_counts[symbol] = 0
        self.symbol_prices[symbol] = None
        self.symbol_price_history[symbol] = []
        
        self.logger.info(f"Reset statistics for symbol {symbol}")
        return True
    
    def save_price_history_to_csv(self, symbol, filename, count=None):
        """
        Save price history for a symbol to CSV file
        
        Args:
            symbol (str): Trading symbol
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_price_history_dataframe(symbol, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No price history to save for symbol {symbol}")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} price records for {symbol} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save price history for {symbol} to CSV: {str(e)}")
            return False
    
    def save_price_history_to_parquet(self, symbol, filename, count=None):
        """
        Save price history for a symbol to Parquet file
        
        Args:
            symbol (str): Trading symbol
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_price_history_dataframe(symbol, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No price history to save for symbol {symbol}")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} price records for {symbol} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save price history for {symbol} to Parquet: {str(e)}")
            return False
