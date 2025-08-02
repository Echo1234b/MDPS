# time_drift_monitor.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import logging
import ntplib
import pytz

class TimeDriftMonitor:
    """
    A class to monitor time drift between MT5 platform time and actual exchange time.
    Logs and alerts on significant time discrepancies.
    """
    
    def __init__(self, time_sync_engine, check_interval=60, drift_threshold=1.0):
        """
        Initialize Time Drift Monitor
        
        Args:
            time_sync_engine (TimeSyncEngine): Time sync engine instance
            check_interval (int): Check interval in seconds
            drift_threshold (float): Drift threshold in seconds for alerts
        """
        self.time_sync_engine = time_sync_engine
        self.check_interval = check_interval
        self.drift_threshold = drift_threshold
        self.logger = self._setup_logger()
        
        # Monitoring status
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Drift history
        self.drift_history = []
        self.max_history_size = 1000
        
        # Alert callbacks
        self.alert_callbacks = []
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("TimeDriftMonitor")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("time_drift_monitor.log")
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
    
    def add_alert_callback(self, callback):
        """
        Add alert callback function
        
        Args:
            callback (function): Callback function to call on alert
        """
        self.alert_callbacks.append(callback)
        self.logger.info("Added alert callback")
    
    def remove_alert_callback(self, callback):
        """
        Remove alert callback function
        
        Args:
            callback (function): Callback function to remove
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.info("Removed alert callback")
    
    def set_check_interval(self, interval):
        """
        Set check interval
        
        Args:
            interval (int): Check interval in seconds
        """
        self.check_interval = interval
        self.logger.info(f"Set check interval to {interval} seconds")
    
    def set_drift_threshold(self, threshold):
        """
        Set drift threshold for alerts
        
        Args:
            threshold (float): Drift threshold in seconds
        """
        self.drift_threshold = threshold
        self.logger.info(f"Set drift threshold to {threshold} seconds")
    
    def start_monitoring(self):
        """
        Start monitoring time drift
        
        Returns:
            bool: Whether successfully started monitoring
        """
        if self.is_monitoring:
            self.logger.warning("Time drift monitoring is already running")
            return False
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_drift)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Started time drift monitoring")
        return True
    
    def stop_monitoring(self):
        """
        Stop monitoring time drift
        
        Returns:
            bool: Whether successfully stopped monitoring
        """
        if not self.is_monitoring:
            self.logger.warning("Time drift monitoring is not running")
            return False
        
        # Stop monitoring thread
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Stopped time drift monitoring")
        return True
    
    def _monitor_drift(self):
        """
        Internal method to monitor time drift, runs in separate thread
        """
        while self.is_monitoring:
            try:
                # Get current time drift
                drift = self.time_sync_engine.mt5_to_ntp_drift
                
                # Record drift
                self._record_drift(drift)
                
                # Check if drift exceeds threshold
                if abs(drift) > self.drift_threshold:
                    self._trigger_alert(drift)
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring time drift: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _record_drift(self, drift):
        """
        Record time drift
        
        Args:
            drift (float): Time drift in seconds
        """
        # Create drift record
        record = {
            'time': datetime.now(),
            'drift': drift
        }
        
        # Add to history
        self.drift_history.append(record)
        
        # Limit history size
        if len(self.drift_history) > self.max_history_size:
            self.drift_history.pop(0)
        
        # Log
        self.logger.debug(f"Recorded time drift: {drift:.3f} seconds")
    
    def _trigger_alert(self, drift):
        """
        Trigger alert for significant time drift
        
        Args:
            drift (float): Time drift in seconds
        """
        # Create alert message
        message = f"Significant time drift detected: {drift:.3f} seconds"
        
        # Log alert
        self.logger.warning(message)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(drift, message)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")
    
    def get_drift_history(self, count=None):
        """
        Get drift history
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of drift records
        """
        if count is None:
            return self.drift_history.copy()
        else:
            return self.drift_history[-count:]
    
    def get_drift_dataframe(self, count=None):
        """
        Get drift history as DataFrame
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing drift history
        """
        history = self.get_drift_history(count)
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    def get_drift_stats(self):
        """
        Get drift statistics
        
        Returns:
            dict: Drift statistics
        """
        if not self.drift_history:
            return {}
        
        drifts = [record['drift'] for record in self.drift_history]
        
        return {
            'min': min(drifts),
            'max': max(drifts),
            'avg': sum(drifts) / len(drifts),
            'current': drifts[-1] if drifts else None,
            'abs_avg': sum(abs(d) for d in drifts) / len(drifts),
            'abs_max': max(abs(d) for d in drifts),
            'samples': len(drifts)
        }
    
    def save_drift_to_csv(self, filename, count=None):
        """
        Save drift history to CSV file
        
        Args:
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_drift_dataframe(count)
            
            if df.empty:
                self.logger.warning("No drift data to save")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} drift records to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save drift data to CSV: {str(e)}")
            return False
    
    def save_drift_to_parquet(self, filename, count=None):
        """
        Save drift history to Parquet file
        
        Args:
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_drift_dataframe(count)
            
            if df.empty:
                self.logger.warning("No drift data to save")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} drift records to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save drift data to Parquet: {str(e)}")
            return False
    
    def clear_drift_history(self):
        """
        Clear drift history
        
        Returns:
            int: Number of records cleared
        """
        count = len(self.drift_history)
        self.drift_history.clear()
        self.logger.info(f"Cleared {count} drift records")
        return count
