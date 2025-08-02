# live_price_feed.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import threading
import queue
import logging
import json

class LivePriceFeed:
    """
    A class to stream live price quotes, OHLC updates, and instrument status.
    Provides normalized price feed for downstream consumption.
    """
    
    def __init__(self, mt5_connector, buffer_size=10000):
        """
        Initialize Live Price Feed
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            buffer_size (int): Memory buffer size
        """
        self.mt5_connector = mt5_connector
        self.buffer_size = buffer_size
        self.price_buffer = queue.Queue(maxsize=buffer_size)
        self.is_feeding = False
        self.feed_thread = None
        self.logger = self._setup_logger()
        self.subscribers = {}
        self.symbols = set()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("LivePriceFeed")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("live_price_feed.log")
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
    
    def subscribe(self, subscriber_id, callback, symbols=None):
        """
        Subscribe to price feed
        
        Args:
            subscriber_id (str): Unique subscriber identifier
            callback (function): Callback function to handle price updates
            symbols (list): List of symbols to subscribe to, None means all symbols
            
        Returns:
            bool: Whether successfully subscribed
        """
        if subscriber_id in self.subscribers:
            self.logger.warning(f"Subscriber {subscriber_id} already exists")
            return False
        
        self.subscribers[subscriber_id] = {
            'callback': callback,
            'symbols': set(symbols) if symbols else None
        }
        
        # Update symbols set
        if symbols:
            self.symbols.update(symbols)
        
        self.logger.info(f"Added subscriber {subscriber_id}")
        return True
    
    def unsubscribe(self, subscriber_id):
        """
        Unsubscribe from price feed
        
        Args:
            subscriber_id (str): Subscriber identifier
            
        Returns:
            bool: Whether successfully unsubscribed
        """
        if subscriber_id not in self.subscribers:
            self.logger.warning(f"Subscriber {subscriber_id} not found")
            return False
        
        # Remove subscriber
        del self.subscribers[subscriber_id]
        
        # Update symbols set if needed
        symbols_in_use = set()
        for sub in self.subscribers.values():
            if sub['symbols']:
                symbols_in_use.update(sub['symbols'])
        
        # If no subscribers are using specific symbols, clear the symbols set
        if not symbols_in_use:
            self.symbols = set()
        else:
            self.symbols = symbols_in_use
        
        self.logger.info(f"Removed subscriber {subscriber_id}")
        return True
    
    def start_feed(self):
        """
        Start price feed
        
        Returns:
            bool: Whether successfully started feed
        """
        if self.is_feeding:
            self.logger.warning("Price feed is already running")
            return False
        
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
