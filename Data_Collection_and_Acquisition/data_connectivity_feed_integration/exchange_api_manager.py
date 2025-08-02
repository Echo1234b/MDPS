# exchange_api_manager.py
import ccxt
import asyncio
import aiohttp
import requests
from datetime import datetime
import pandas as pd
import time
import logging

class ExchangeAPIManager:
    """
    A class to manage multiple exchange (Binance, Coinbase, OANDA, etc.) REST/WebSocket connections.
    Standardizes API authentication, rate limiting, reconnection, and error handling.
    """
    
    def __init__(self):
        """Initialize exchange API manager"""
        self.exchanges = {}
        self.sessions = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("ExchangeAPIManager")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("exchange_api_manager.log")
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
    
    def add_exchange(self, exchange_id, api_key=None, secret=None, options=None):
        """
        Add exchange connection
        
        Args:
            exchange_id (str): Exchange ID (e.g., 'binance', 'coinbase')
            api_key (str): API key
            secret (str): API secret
            options (dict): Exchange-specific options
            
        Returns:
            bool: Whether successfully added
        """
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'options': options or {}
            })
            
            # Enable rate limiting
            exchange.enableRateLimit = True
            
            # Store exchange instance
            self.exchanges[exchange_id] = exchange
            
            # Create session
            self.sessions[exchange_id] = requests.Session()
            
            self.logger.info(f"Added exchange: {exchange_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add exchange {exchange_id}: {str(e)}")
            return False
    
    def remove_exchange(self, exchange_id):
        """
        Remove exchange connection
        
        Args:
            exchange_id (str): Exchange ID
            
        Returns:
            bool: Whether successfully removed
        """
        if exchange_id in self.exchanges:
            # Close exchange connection
            if hasattr(self.exchanges[exchange_id], 'close'):
                self.exchanges[exchange_id].close()
            
            # Close session
            if exchange_id in self.sessions:
                self.sessions[exchange_id].close()
                del self.sessions[exchange_id]
            
            # Remove from dictionary
            del self.exchanges[exchange_id]
            
            self.logger.info(f"Removed exchange: {exchange_id}")
            return True
        else:
            self.logger.warning(f"Exchange not found: {exchange_id}")
            return False
    
    def get_exchange(self, exchange_id):
        """
        Get exchange instance
        
        Args:
            exchange_id (str): Exchange ID
            
        Returns:
            ccxt.Exchange: Exchange instance
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange not found: {exchange_id}")
            return None
        return self.exchanges[exchange_id]
    
    def fetch_ohlcv(self, exchange_id, symbol, timeframe, since=None, limit=None):
        """
        Get OHLCV data
        
        Args:
            exchange_id (str): Exchange ID
            symbol (str): Trading symbol
            timeframe (str): Timeframe (e.g., '1m', '5m', '1h', '1d')
            since (int): Timestamp (ms) to start getting data from
            limit (int): Number of data points to get
            
        Returns:
            pandas.DataFrame: DataFrame containing OHLCV data
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange not found: {exchange_id}")
            return None
        
        try:
            exchange = self.exchanges[exchange_id]
            
            # Get OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch OHLCV from {exchange_id}: {str(e)}")
            return None
    
    def fetch_ticker(self, exchange_id, symbol):
        """
        Get ticker data
        
        Args:
            exchange_id (str): Exchange ID
            symbol (str): Trading symbol
            
        Returns:
            dict: Ticker data
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange not found: {exchange_id}")
            return None
        
        try:
            exchange = self.exchanges[exchange_id]
            return exchange.fetch_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker from {exchange_id}: {str(e)}")
            return None
    
    def fetch_order_book(self, exchange_id, symbol, limit=None):
        """
        Get order book data
        
        Args:
            exchange_id (str): Exchange ID
            symbol (str): Trading symbol
            limit (int): Order book depth to get
            
        Returns:
            dict: Order book data
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange not found: {exchange_id}")
            return None
        
        try:
            exchange = self.exchanges[exchange_id]
            return exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            self.logger.error(f"Failed to fetch order book from {exchange_id}: {str(e)}")
            return None
    
    def fetch_balance(self, exchange_id):
        """
        Get account balance
        
        Args:
            exchange_id (str): Exchange ID
            
        Returns:
            dict: Account balance data
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange not found: {exchange_id}")
            return None
        
        try:
            exchange = self.exchanges[exchange_id]
            return exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"Failed to fetch balance from {exchange_id}: {str(e)}")
            return None
    
    async def create_websocket_connection(self, exchange_id, channels, callback):
        """
        Create WebSocket connection
        
        Args:
            exchange_id (str): Exchange ID
            channels (list): List of channels to subscribe to
            callback (function): Callback function to process received data
            
        Returns:
            bool: Whether successfully created connection
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange not found: {exchange_id}")
            return False
        
        try:
            exchange = self.exchanges[exchange_id]
            
            # Check if exchange supports WebSocket
            if not hasattr(exchange, 'watch'):
                self.logger.error(f"Exchange {exchange_id} does not support WebSocket")
                return False
            
            # Create WebSocket connection
            async with aiohttp.ClientSession() as session:
                for channel in channels:
                    try:
                        while True:
                            # Watch channel
                            data = await exchange.watch(channel)
                            # Call callback function to process data
                            await callback(data)
                    except Exception as e:
                        self.logger.error(f"WebSocket error for {channel}: {str(e)}")
                        # Wait for a while before retrying
                        await asyncio.sleep(5)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to create WebSocket connection for {exchange_id}: {str(e)}")
            return False
    
    def get_exchange_status(self, exchange_id):
        """
        Get exchange status
        
        Args:
            exchange_id (str): Exchange ID
            
        Returns:
            dict: Exchange status information
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange not found: {exchange_id}")
            return None
        
        try:
            exchange = self.exchanges[exchange_id]
            status = {
                'id': exchange.id,
                'name': exchange.name,
                'countries': exchange.countries,
                'urls': exchange.urls,
                'version': exchange.version,
                'api': exchange.api,
                'has': exchange.has,
                'timeframes': exchange.timeframes,
                'timeout': exchange.timeout,
                'rateLimit': exchange.rateLimit,
                'connected': True  # If no exception, consider connected
            }
            return status
        except Exception as e:
            self.logger.error(f"Failed to get status for {exchange_id}: {str(e)}")
            return {'id': exchange_id, 'connected': False, 'error': str(e)}
    
    def __del__(self):
        """Destructor, ensure all connections are closed"""
        for exchange_id in list(self.exchanges.keys()):
            self.remove_exchange(exchange_id)
