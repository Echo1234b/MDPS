# metatrader5_connector.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import logging

class MetaTrader5Connector:
    """
    A class to interface with MetaTrader 5 terminal, providing real-time/historical 
    data access, account information, and order access.
    Uses MetaTrader5 Python package to enable Python-based broker data stream access.
    """
    
    def __init__(self):
        """Initialize MT5 connection"""
        self.connected = False
        self.account_info = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("MetaTrader5Connector")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("mt5_connector.log")
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
    
    def connect(self):
        """Establish connection to MT5 terminal"""
        if not mt5.initialize():
            self.logger.error("initialize() failed, error code =", mt5.last_error())
            return False
        
        self.connected = True
        self.logger.info("Connected to MetaTrader 5")
        self.account_info = mt5.account_info()
        if self.account_info is not None:
            self.logger.info(f"Account: {self.account_info.login}, Balance: {self.account_info.balance}")
        return True
    
    def disconnect(self):
        """Close connection to MT5 terminal"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MetaTrader 5")
    
    def get_account_info(self):
        """Get account information"""
        if not self.connected:
            if not self.connect():
                return None
        return mt5.account_info()
    
    def get_symbol_info(self, symbol):
        """Get symbol information"""
        if not self.connected:
            if not self.connect():
                return None
        return mt5.symbol_info(symbol)
    
    def get_ticks(self, symbol, start_date, end_date):
        """
        Get tick data for specified date range
        
        Args:
            symbol (str): Trading symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            pandas.DataFrame: DataFrame containing tick data
        """
        if not self.connected:
            if not self.connect():
                return None
        
        ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
        if ticks is None:
            self.logger.error("Failed to get ticks, error code =", mt5.last_error())
            return None
        
        ticks_frame = pd.DataFrame(ticks)
        ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s')
        return ticks_frame
    
    def get_rates(self, symbol, timeframe, start_date, end_date):
        """
        Get OHLCV data for specified date range
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): Timeframe (MT5 timeframe constant)
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            pandas.DataFrame: DataFrame containing OHLCV data
        """
        if not self.connected:
            if not self.connect():
                return None
        
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None:
            self.logger.error("Failed to get rates, error code =", mt5.last_error())
            return None
        
        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        return rates_frame
    
    def get_last_n_rates(self, symbol, timeframe, n):
        """
        Get last n OHLCV data
        
        Args:
            symbol (str): Trading symbol
            timeframe (int): Timeframe (MT5 timeframe constant)
            n (int): Number of bars to get
            
        Returns:
            pandas.DataFrame: DataFrame containing OHLCV data
        """
        if not self.connected:
            if not self.connect():
                return None
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        if rates is None:
            self.logger.error("Failed to get rates, error code =", mt5.last_error())
            return None
        
        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        return rates_frame
    
    def place_order(self, order_type, symbol, volume, price=None, stop_loss=None, take_profit=None, comment=""):
        """
        Place order
        
        Args:
            order_type (int): Order type (MT5 order type constant)
            symbol (str): Trading symbol
            volume (float): Trading volume
            price (float): Price, can be None for market order
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            comment (str): Order comment
            
        Returns:
            dict: Order result
        """
        if not self.connected:
            if not self.connect():
                return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error("Symbol not found:", symbol)
            return None
        
        if price is None:
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
            "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error("order_send failed, retcode =", result.retcode)
            return None
        
        return result
    
    def __del__(self):
        """Destructor, ensure connection is closed"""
        self.disconnect()
