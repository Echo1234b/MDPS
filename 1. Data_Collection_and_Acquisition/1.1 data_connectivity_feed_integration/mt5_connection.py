"""
MetaTrader 5 Connection Module

Handles MetaTrader 5 terminal integration with full API coverage,
real-time price feeds, order execution, and account management.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

class MT5Status(Enum):
    DISCONNECTED = 0
    CONNECTED = 1
    CONNECTING = 2
    ERROR = 3

@dataclass
class MT5Config:
    """MT5 connection configuration"""
    server: str
    login: int
    password: str
    path: Optional[str] = None
    timeout: int = 60000
    portable: bool = False

@dataclass
class SymbolInfo:
    """Symbol information from MT5"""
    name: str
    description: str
    currency_base: str
    currency_profit: str
    currency_margin: str
    digits: int
    point: float
    spread: int
    stops_level: int
    lot_size: float
    min_lot: float
    max_lot: float
    lot_step: float
    margin_initial: float
    margin_maintenance: float
    session_deals: int
    session_buy_orders: int
    session_sell_orders: int
    volume: int
    volumehigh: int
    volumelow: int
    time: datetime
    bid: float
    ask: float
    last: float
    volume_real: float

class MT5Connection:
    """
    MetaTrader 5 terminal integration with comprehensive API coverage
    """
    
    def __init__(self, config: MT5Config):
        """
        Initialize MT5 connection
        
        Args:
            config: MT5 connection configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.status = MT5Status.DISCONNECTED
        self.account_info = None
        self.symbols_info = {}
        self.price_callbacks = {}
        self.monitoring_thread = None
        self.running = False
        self.last_error = None
        
    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal
        
        Returns:
            bool: Connection success status
        """
        try:
            self.status = MT5Status.CONNECTING
            self.logger.info("Connecting to MT5 terminal...")
            
            # Initialize MT5 connection
            if self.config.path:
                if not mt5.initialize(path=self.config.path, 
                                    login=self.config.login,
                                    password=self.config.password,
                                    server=self.config.server,
                                    timeout=self.config.timeout,
                                    portable=self.config.portable):
                    error = mt5.last_error()
                    self.last_error = f"MT5 initialization failed: {error}"
                    self.logger.error(self.last_error)
                    self.status = MT5Status.ERROR
                    return False
            else:
                if not mt5.initialize():
                    error = mt5.last_error()
                    self.last_error = f"MT5 initialization failed: {error}"
                    self.logger.error(self.last_error)
                    self.status = MT5Status.ERROR
                    return False
                
                # Login to account
                if not mt5.login(login=self.config.login,
                               password=self.config.password,
                               server=self.config.server):
                    error = mt5.last_error()
                    self.last_error = f"MT5 login failed: {error}"
                    self.logger.error(self.last_error)
                    self.status = MT5Status.ERROR
                    return False
            
            # Get account information
            self.account_info = mt5.account_info()
            if self.account_info is None:
                error = mt5.last_error()
                self.last_error = f"Failed to get account info: {error}"
                self.logger.error(self.last_error)
                self.status = MT5Status.ERROR
                return False
            
            self.status = MT5Status.CONNECTED
            self.logger.info(f"Successfully connected to MT5. Account: {self.account_info.login}")
            
            # Start monitoring thread
            self.start_monitoring()
            
            return True
            
        except Exception as e:
            self.last_error = f"Connection error: {str(e)}"
            self.logger.error(self.last_error)
            self.status = MT5Status.ERROR
            return False
    
    def disconnect(self):
        """Disconnect from MT5 terminal"""
        try:
            self.stop_monitoring()
            mt5.shutdown()
            self.status = MT5Status.DISCONNECTED
            self.logger.info("Disconnected from MT5 terminal")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {str(e)}")
    
    def get_symbols(self, group: Optional[str] = None) -> List[str]:
        """
        Get available symbols
        
        Args:
            group: Symbol group filter (e.g., "*USD*", "Forex*")
            
        Returns:
            List[str]: List of symbol names
        """
        try:
            if group:
                symbols = mt5.symbols_get(group=group)
            else:
                symbols = mt5.symbols_get()
            
            if symbols is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get symbols: {error}")
                return []
            
            return [symbol.name for symbol in symbols]
            
        except Exception as e:
            self.logger.error(f"Error getting symbols: {str(e)}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get detailed symbol information
        
        Args:
            symbol: Symbol name
            
        Returns:
            SymbolInfo: Symbol information or None if failed
        """
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get symbol info for {symbol}: {error}")
                return None
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                tick_time = datetime.now()
                bid = ask = last = 0.0
                volume_real = 0.0
            else:
                tick_time = datetime.fromtimestamp(tick.time)
                bid = tick.bid
                ask = tick.ask
                last = tick.last
                volume_real = tick.volume_real
            
            symbol_info = SymbolInfo(
                name=info.name,
                description=info.description,
                currency_base=info.currency_base,
                currency_profit=info.currency_profit,
                currency_margin=info.currency_margin,
                digits=info.digits,
                point=info.point,
                spread=info.spread,
                stops_level=info.stops_level,
                lot_size=info.trade_contract_size,
                min_lot=info.volume_min,
                max_lot=info.volume_max,
                lot_step=info.volume_step,
                margin_initial=info.margin_initial,
                margin_maintenance=info.margin_maintenance,
                session_deals=info.session_deals,
                session_buy_orders=info.session_buy_orders,
                session_sell_orders=info.session_sell_orders,
                volume=info.volume,
                volumehigh=info.volumehigh,
                volumelow=info.volumelow,
                time=tick_time,
                bid=bid,
                ask=ask,
                last=last,
                volume_real=volume_real
            )
            
            self.symbols_info[symbol] = symbol_info
            return symbol_info
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return None
    
    def get_rates(self, symbol: str, timeframe: int, start: datetime, count: int) -> Optional[pd.DataFrame]:
        """
        Get historical rates (OHLCV data)
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant (mt5.TIMEFRAME_M1, etc.)
            start: Start datetime
            count: Number of bars
            
        Returns:
            pd.DataFrame: OHLCV data or None if failed
        """
        try:
            rates = mt5.copy_rates_from(symbol, timeframe, start, count)
            if rates is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get rates for {symbol}: {error}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'TickVolume', 'Spread', 'RealVolume']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting rates for {symbol}: {str(e)}")
            return None
    
    def get_rates_range(self, symbol: str, timeframe: int, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """
        Get historical rates for a date range
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            start: Start datetime
            end: End datetime
            
        Returns:
            pd.DataFrame: OHLCV data or None if failed
        """
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start, end)
            if rates is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get rates range for {symbol}: {error}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'TickVolume', 'Spread', 'RealVolume']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting rates range for {symbol}: {str(e)}")
            return None
    
    def get_ticks(self, symbol: str, start: datetime, count: int) -> Optional[pd.DataFrame]:
        """
        Get tick data
        
        Args:
            symbol: Symbol name
            start: Start datetime
            count: Number of ticks
            
        Returns:
            pd.DataFrame: Tick data or None if failed
        """
        try:
            ticks = mt5.copy_ticks_from(symbol, start, count, mt5.COPY_TICKS_ALL)
            if ticks is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get ticks for {symbol}: {error}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ticks)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
                df.set_index('time_msc', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting ticks for {symbol}: {str(e)}")
            return None
    
    def get_current_tick(self, symbol: str) -> Optional[Dict]:
        """
        Get current tick for symbol
        
        Args:
            symbol: Symbol name
            
        Returns:
            Dict: Current tick data or None if failed
        """
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to get current tick for {symbol}: {error}")
                return None
            
            return {
                'symbol': symbol,
                'time': datetime.fromtimestamp(tick.time),
                'time_msc': datetime.fromtimestamp(tick.time_msc / 1000),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'volume_real': tick.volume_real,
                'flags': tick.flags
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current tick for {symbol}: {str(e)}")
            return None
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List[Dict]: List of positions
        """
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'time': datetime.fromtimestamp(pos.time),
                    'time_msc': datetime.fromtimestamp(pos.time_msc / 1000),
                    'time_update': datetime.fromtimestamp(pos.time_update),
                    'time_update_msc': datetime.fromtimestamp(pos.time_update_msc / 1000),
                    'type': pos.type,
                    'magic': pos.magic,
                    'identifier': pos.identifier,
                    'reason': pos.reason,
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'price_current': pos.price_current,
                    'swap': pos.swap,
                    'profit': pos.profit,
                    'symbol': pos.symbol,
                    'comment': pos.comment,
                    'external_id': pos.external_id
                })
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get pending orders
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List[Dict]: List of orders
        """
        try:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
            
            if orders is None:
                return []
            
            order_list = []
            for order in orders:
                order_list.append({
                    'ticket': order.ticket,
                    'time_setup': datetime.fromtimestamp(order.time_setup),
                    'time_setup_msc': datetime.fromtimestamp(order.time_setup_msc / 1000),
                    'time_expiration': datetime.fromtimestamp(order.time_expiration) if order.time_expiration > 0 else None,
                    'type': order.type,
                    'type_time': order.type_time,
                    'type_filling': order.type_filling,
                    'state': order.state,
                    'magic': order.magic,
                    'position_id': order.position_id,
                    'position_by_id': order.position_by_id,
                    'reason': order.reason,
                    'volume_initial': order.volume_initial,
                    'volume_current': order.volume_current,
                    'price_open': order.price_open,
                    'sl': order.sl,
                    'tp': order.tp,
                    'price_current': order.price_current,
                    'price_stoplimit': order.price_stoplimit,
                    'symbol': order.symbol,
                    'comment': order.comment,
                    'external_id': order.external_id
                })
            
            return order_list
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def send_order(self, symbol: str, order_type: int, volume: float, 
                   price: Optional[float] = None, sl: Optional[float] = None, 
                   tp: Optional[float] = None, comment: str = "", 
                   magic: int = 0) -> Optional[Dict]:
        """
        Send a trading order
        
        Args:
            symbol: Symbol name
            order_type: Order type (mt5.ORDER_TYPE_BUY, etc.)
            volume: Order volume
            price: Order price (for pending orders)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            magic: Magic number
            
        Returns:
            Dict: Order result or None if failed
        """
        try:
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if price is not None:
                request["price"] = price
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to send order: {error}")
                return None
            
            return {
                'retcode': result.retcode,
                'deal': result.deal,
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'bid': result.bid,
                'ask': result.ask,
                'comment': result.comment,
                'request_id': result.request_id,
                'retcode_external': result.retcode_external
            }
            
        except Exception as e:
            self.logger.error(f"Error sending order: {str(e)}")
            return None
    
    def close_position(self, position_id: int) -> Optional[Dict]:
        """
        Close a position
        
        Args:
            position_id: Position ID to close
            
        Returns:
            Dict: Close result or None if failed
        """
        try:
            # Get position info
            positions = mt5.positions_get(ticket=position_id)
            if not positions:
                self.logger.error(f"Position {position_id} not found")
                return None
            
            position = positions[0]
            
            # Determine opposite order type
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
            else:
                order_type = mt5.ORDER_TYPE_BUY
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position_id,
                "comment": f"Close position {position_id}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            if result is None:
                error = mt5.last_error()
                self.logger.error(f"Failed to close position: {error}")
                return None
            
            return {
                'retcode': result.retcode,
                'deal': result.deal,
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'bid': result.bid,
                'ask': result.ask,
                'comment': result.comment,
                'request_id': result.request_id,
                'retcode_external': result.retcode_external
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return None
    
    def start_monitoring(self):
        """Start connection monitoring"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_connection)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.info("Started MT5 connection monitoring")
    
    def stop_monitoring(self):
        """Stop connection monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Stopped MT5 connection monitoring")
    
    def _monitor_connection(self):
        """Monitor MT5 connection health"""
        while self.running:
            try:
                # Check connection by getting account info
                account_info = mt5.account_info()
                if account_info is None:
                    self.status = MT5Status.ERROR
                    self.last_error = "Lost connection to MT5"
                    self.logger.error("Lost connection to MT5 terminal")
                else:
                    if self.status == MT5Status.ERROR:
                        self.status = MT5Status.CONNECTED
                        self.logger.info("MT5 connection restored")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.status = MT5Status.ERROR
                self.last_error = f"Monitoring error: {str(e)}"
                self.logger.error(f"MT5 monitoring error: {str(e)}")
                time.sleep(10)
    
    def is_connected(self) -> bool:
        """Check if connected to MT5"""
        return self.status == MT5Status.CONNECTED
    
    def get_status(self) -> Dict:
        """Get connection status information"""
        return {
            'status': self.status.name,
            'account_info': self.account_info._asdict() if self.account_info else None,
            'last_error': self.last_error,
            'symbols_count': len(self.symbols_info)
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.disconnect()