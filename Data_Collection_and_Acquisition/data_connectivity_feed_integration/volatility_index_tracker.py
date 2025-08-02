# volatility_index_tracker.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
import logging
import requests
import json

class VolatilityIndexTracker:
    """
    A class to interface with APIs to get VIX or other volatility indices.
    Tracks and stores historical volatility measurements.
    """
    
    def __init__(self, buffer_size=10000):
        """
        Initialize Volatility Index Tracker
        
        Args:
            buffer_size (int): Memory buffer size
        """
        self.buffer_size = buffer_size
        self.volatility_buffer = queue.Queue(maxsize=buffer_size)
        self.is_tracking = False
        self.tracking_thread = None
        self.logger = self._setup_logger()
        self.api_endpoints = {
            'vix': 'https://www.cboe.com/json/vix',
            'vix3m': 'https://www.cboe.com/json/vix3m',
            'vix6m': 'https://www.cboe.com/json/vix6m',
            'vix9m': 'https://www.cboe.com/json/vix9m',
            'vix1y': 'https://www.cboe.com/json/vix1y',
            'gvz': 'https://www.cboe.com/json/gvz',
            'ovx': 'https://www.cboe.com/json/ovx'
        }
        self.volatility_history = {}
        self.update_interval = 60  # seconds
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("VolatilityIndexTracker")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("volatility_index_tracker.log")
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
    
    def add_api_endpoint(self, name, url):
        """
        Add custom API endpoint for volatility index
        
        Args:
            name (str): Name of the volatility index
            url (str): URL of the API endpoint
        """
        self.api_endpoints[name] = url
        self.logger.info(f"Added API endpoint for {name}: {url}")
    
    def set_update_interval(self, interval):
        """
        Set update interval for tracking
        
        Args:
            interval (int): Update interval in seconds
        """
        self.update_interval = interval
        self.logger.info(f"Set update interval to {interval} seconds")
    
    def start_tracking(self, indices=None):
        """
        Start tracking volatility indices
        
        Args:
            indices (list): List of indices to track, None means track all
            
        Returns:
            bool: Whether successfully started tracking
        """
        if self.is_tracking:
            self.logger.warning("Volatility tracking is already running")
            return False
        
        # Determine which indices to track
        if indices is None:
            indices = list(self.api_endpoints.keys())
        else:
            # Validate indices
            invalid_indices = [idx for idx in indices if idx not in self.api_endpoints]
            if invalid_indices:
                self.logger.error(f"Invalid indices: {invalid_indices}")
                return False
        
        # Initialize history for each index
        for index in indices:
            if index not in self.volatility_history:
                self.volatility_history[index] = []
        
        # Start tracking thread
        self.is_tracking = True
        self.tracking_thread = threading.Thread(
            target=self._track_volatility, 
            args=(indices,)
        )
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        self.logger.info(f"Started tracking volatility indices: {indices}")
        return True
    
    def stop_tracking(self):
        """
        Stop tracking volatility indices
        
        Returns:
            bool: Whether successfully stopped tracking
        """
        if not self.is_tracking:
            self.logger.warning("Volatility tracking is not running")
            return False
        
        # Stop tracking thread
        self.is_tracking = False
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=5)
        
        self.logger.info("Stopped volatility tracking")
        return True
    
    def _track_volatility(self, indices):
        """
        Internal method to track volatility indices, runs in separate thread
        
        Args:
            indices (list): List of indices to track
        """
        while self.is_tracking:
            try:
                # Get current time
                current_time = datetime.now()
                
                # Track each index
                for index in indices:
                    if index not in self.api_endpoints:
                        continue
                    
                    try:
                        # Get data from API
                        url = self.api_endpoints[index]
                        response = requests.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            # Parse JSON response
                            data = response.json()
                            
                            # Extract value based on index type
                            value = None
                            if index == 'vix':
                                value = data.get('vix', {}).get('current')
                            elif index == 'vix3m':
                                value = data.get('vix3m', {}).get('current')
                            elif index == 'vix6m':
                                value = data.get('vix6m', {}).get('current')
                            elif index == 'vix9m':
                                value = data.get('vix9m', {}).get('current')
                            elif index == 'vix1y':
                                value = data.get('vix1y', {}).get('current')
                            elif index == 'gvz':
                                value = data.get('gvz', {}).get('current')
                            elif index == 'ovx':
                                value = data.get('ovx', {}).get('current')
                            else:
                                # Try to extract value from common fields
                                if 'current' in data:
                                    value = data['current']
                                elif 'value' in data:
                                    value = data['value']
                                elif 'price' in data:
                                    value = data['price']
                            
                            if value is not None:
                                try:
                                    value = float(value)
                                except (ValueError, TypeError):
                                    self.logger.warning(f"Invalid value for {index}: {value}")
                                    continue
                                
                                # Create volatility data dictionary
                                volatility_data = {
                                    'time': current_time,
                                    'index': index,
                                    'value': value
                                }
                                
                                # Put volatility data into buffer
                                if self.volatility_buffer.full():
                                    # If buffer is full, remove oldest data
                                    try:
                                        self.volatility_buffer.get_nowait()
                                    except queue.Empty:
                                        pass
                                
                                self.volatility_buffer.put(volatility_data)
                                
                                # Update history
                                self.volatility_history[index].append({
                                    'time': current_time,
                                    'value': value
                                })
                                
                                # Keep only recent history (last 1000 entries)
                                if len(self.volatility_history[index]) > 1000:
                                    self.volatility_history[index].pop(0)
                                
                                # Log
                                self.logger.debug(f"Updated {index}: {value}")
                            else:
                                self.logger.warning(f"No value found for {index}")
                        else:
                            self.logger.error(f"API request failed for {index}: {response.status_code}")
                    
                    except Exception as e:
                        self.logger.error(f"Error tracking {index}: {str(e)}")
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in volatility tracking thread: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def get_volatility_data(self, count=None):
        """
        Get volatility data from buffer
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of volatility data
        """
        data = []
        
        if count is None:
            # Get all data
            while not self.volatility_buffer.empty():
                try:
                    data.append(self.volatility_buffer.get_nowait())
                except queue.Empty:
                    break
        else:
            # Get specified number of records
            for _ in range(min(count, self.volatility_buffer.qsize())):
                try:
                    data.append(self.volatility_buffer.get_nowait())
                except queue.Empty:
                    break
        
        return data
    
    def get_volatility_dataframe(self, count=None):
        """
        Get volatility data from buffer and convert to DataFrame
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing volatility data
        """
        data = self.get_volatility_data(count)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    def get_latest_value(self, index):
        """
        Get latest value for a volatility index
        
        Args:
            index (str): Name of the volatility index
            
        Returns:
            float: Latest value or None if not available
        """
        if index not in self.volatility_history or not self.volatility_history[index]:
            return None
        
        return self.volatility_history[index][-1]['value']
    
    def get_volatility_stats(self, index):
        """
        Get volatility statistics for an index
        
        Args:
            index (str): Name of the volatility index
            
        Returns:
            dict: Volatility statistics
        """
        if index not in self.volatility_history or not self.volatility_history[index]:
            return {}
        
        values = [entry['value'] for entry in self.volatility_history[index]]
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'current': values[-1] if values else None,
            'samples': len(values),
            'std_dev': np.std(values),
            'percentile_25': np.percentile(values, 25),
            'percentile_75': np.percentile(values, 75)
        }
    
    def save_volatility_to_csv(self, filename, count=None):
        """
        Save volatility data in buffer to CSV file
        
        Args:
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_volatility_dataframe(count)
            
            if df.empty:
                self.logger.warning("No volatility data to save")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} volatility records to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save volatility data to CSV: {str(e)}")
            return False
    
    def save_volatility_to_parquet(self, filename, count=None):
        """
        Save volatility data in buffer to Parquet file
        
        Args:
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_volatility_dataframe(count)
            
            if df.empty:
                self.logger.warning("No volatility data to save")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} volatility records to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save volatility data
