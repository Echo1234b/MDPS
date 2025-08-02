# time_sync_engine.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import logging
import ntplib
import pytz

class TimeSyncEngine:
    """
    A class to maintain synchronization between local system time, broker/server time, 
    and exchange time. Handles time zone conversions and alignment.
    """
    
    def __init__(self, mt5_connector, ntp_server='pool.ntp.org'):
        """
        Initialize Time Sync Engine
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            ntp_server (str): NTP server for time synchronization
        """
        self.mt5_connector = mt5_connector
        self.ntp_server = ntp_server
        self.logger = self._setup_logger()
        
        # Time drifts in seconds
        self.local_to_ntp_drift = 0
        self.local_to_mt5_drift = 0
        self.mt5_to_ntp_drift = 0
        
        # Time zones
        self.local_timezone = None
        self.mt5_timezone = None
        
        # Sync status
        self.is_synced = False
        self.last_sync_time = None
        self.sync_interval = 3600  # seconds (1 hour)
        
        # Sync thread
        self.sync_thread = None
        self.is_syncing = False
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("TimeSyncEngine")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("time_sync_engine.log")
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
    
    def set_sync_interval(self, interval):
        """
        Set synchronization interval
        
        Args:
            interval (int): Sync interval in seconds
        """
        self.sync_interval = interval
        self.logger.info(f"Set sync interval to {interval} seconds")
    
    def start_sync(self):
        """
        Start time synchronization
        
        Returns:
            bool: Whether successfully started sync
        """
        if self.is_syncing:
            self.logger.warning("Time sync is already running")
            return False
        
        # Initial sync
        if not self.sync_time():
            return False
        
        # Start sync thread
        self.is_syncing = True
        self.sync_thread = threading.Thread(target=self._continuous_sync)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        self.logger.info("Started time synchronization")
        return True
    
    def stop_sync(self):
        """
        Stop time synchronization
        
        Returns:
            bool: Whether successfully stopped sync
        """
        if not self.is_syncing:
            self.logger.warning("Time sync is not running")
            return False
        
        # Stop sync thread
        self.is_syncing = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)
        
        self.logger.info("Stopped time synchronization")
        return True
    
    def _continuous_sync(self):
        """
        Internal method to continuously sync time, runs in separate thread
        """
        while self.is_syncing:
            try:
                # Check if it's time to sync
                current_time = time.time()
                if self.last_sync_time is None or (current_time - self.last_sync_time) >= self.sync_interval:
                    self.sync_time()
                
                # Sleep for a while
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in time sync thread: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def sync_time(self):
        """
        Synchronize time between local system, NTP, and MT5
        
        Returns:
            bool: Whether successfully synced
        """
        try:
            # Get local time
            local_time = time.time()
            self.local_timezone = datetime.now().astimezone().tzinfo
            
            # Get NTP time
            ntp_time = self._get_ntp_time()
            if ntp_time is None:
                self.logger.error("Failed to get NTP time")
                return False
            
            # Get MT5 time
            mt5_time = self._get_mt5_time()
            if mt5_time is None:
                self.logger.error("Failed to get MT5 time")
                return False
            
            # Calculate drifts
            self.local_to_ntp_drift = ntp_time - local_time
            self.local_to_mt5_drift = mt5_time - local_time
            self.mt5_to_ntp_drift = ntp_time - mt5_time
            
            # Update sync status
            self.is_synced = True
            self.last_sync_time = time.time()
            
            # Log sync results
            self.logger.info(f"Time sync completed:")
            self.logger.info(f"  Local to NTP drift: {self.local_to_ntp_drift:.3f} seconds")
            self.logger.info(f"  Local to MT5 drift: {self.local_to_mt5_drift:.3f} seconds")
            self.logger.info(f"  MT5 to NTP drift: {self.mt5_to_ntp_drift:.3f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing time: {str(e)}")
            return False
    
    def _get_ntp_time(self):
        """
        Get time from NTP server
        
        Returns:
            float: NTP time as Unix timestamp or None if failed
        """
        try:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request(self.ntp_server, timeout=10)
            return response.tx_time
        except Exception as e:
            self.logger.error(f"Error getting NTP time: {str(e)}")
            return None
    
    def _get_mt5_time(self):
        """
        Get time from MT5 server
        
        Returns:
            float: MT5 time as Unix timestamp or None if failed
        """
        try:
            # Ensure MT5 is connected
            if not self.mt5_connector.connected:
                if not self.mt5_connector.connect():
                    return None
            
            # Get terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return None
            
            # Get MT5 time
            mt5_time = terminal_info.lasttrade
            
            # Get MT5 timezone
            mt5_timezone = terminal_info.timezone
            if mt5_timezone is not None:
                # Convert timezone offset to pytz timezone
                hours = mt5_timezone // 60
                minutes = mt5_timezone % 60
                self.mt5_timezone = pytz.FixedOffset(hours * 60 + minutes)
            
            return mt5_time
        except Exception as e:
            self.logger.error(f"Error getting MT5 time: {str(e)}")
            return None
    
    def get_local_time(self):
        """
        Get current local time
        
        Returns:
            datetime: Current local time
        """
        return datetime.now()
    
    def get_ntp_time(self):
        """
        Get current NTP time
        
        Returns:
            datetime: Current NTP time
        """
        if not self.is_synced:
            self.logger.warning("Time not synced, using local time")
            return datetime.now()
        
        # Get current local time and apply drift
        local_time = time.time()
        ntp_time = local_time + self.local_to_ntp_drift
        
        return datetime.fromtimestamp(ntp_time)
    
    def get_mt5_time(self):
        """
        Get current MT5 time
        
        Returns:
            datetime: Current MT5 time
        """
        if not self.is_synced:
            self.logger.warning("Time not synced, using local time")
            return datetime.now()
        
        # Get current local time and apply drift
        local_time = time.time()
        mt5_time = local_time + self.local_to_mt5_drift
        
        return datetime.fromtimestamp(mt5_time)
    
    def local_to_ntp(self, local_time):
        """
        Convert local time to NTP time
        
        Args:
            local_time (datetime): Local time
            
        Returns:
            datetime: NTP time
        """
        if not self.is_synced:
            self.logger.warning("Time not synced, returning local time")
            return local_time
        
        # Convert to timestamp
        timestamp = local_time.timestamp()
        
        # Apply drift
        ntp_timestamp = timestamp + self.local_to_ntp_drift
        
        return datetime.fromtimestamp(ntp_timestamp)
    
    def local_to_mt5(self, local_time):
        """
        Convert local time to MT5 time
        
        Args:
            local_time (datetime): Local time
            
        Returns:
            datetime: MT5 time
        """
        if not self.is_synced:
            self.logger.warning("Time not synced, returning local time")
           
