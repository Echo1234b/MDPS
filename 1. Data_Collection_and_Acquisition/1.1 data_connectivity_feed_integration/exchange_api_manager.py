"""
Exchange API Manager

Multi-exchange API integration with automatic failover, rate limiting,
and connection management for real-time data feeds.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import ccxt
import ccxt.pro as ccxtpro
import threading
import time
from dataclasses import dataclass
from enum import Enum

class ExchangeStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"

@dataclass
class ExchangeConfig:
    """Exchange configuration parameters"""
    name: str
    api_key: str
    secret: str
    sandbox: bool = True
    rate_limit: int = 1000  # requests per minute
    timeout: int = 30000  # milliseconds
    retry_attempts: int = 3
    retry_delay: int = 1000  # milliseconds

@dataclass
class ConnectionHealth:
    """Connection health metrics"""
    exchange: str
    status: ExchangeStatus
    last_ping: Optional[datetime] = None
    latency: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_percentage: float = 100.0

class ExchangeAPIManager:
    """
    Manages connections to multiple cryptocurrency exchanges with
    automatic failover, rate limiting, and health monitoring.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the Exchange API Manager
        
        Args:
            config_file: Path to exchange configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.pro_exchanges: Dict[str, ccxtpro.Exchange] = {}
        self.configs: Dict[str, ExchangeConfig] = {}
        self.health_metrics: Dict[str, ConnectionHealth] = {}
        self.rate_limiters: Dict[str, Dict] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        self.monitoring_thread = None
        
        # Load configurations
        if config_file:
            self._load_config(config_file)
    
    def add_exchange(self, config: ExchangeConfig) -> bool:
        """
        Add an exchange connection
        
        Args:
            config: Exchange configuration
            
        Returns:
            bool: Success status
        """
        try:
            # Initialize REST API exchange
            exchange_class = getattr(ccxt, config.name.lower())
            exchange = exchange_class({
                'apiKey': config.api_key,
                'secret': config.secret,
                'sandbox': config.sandbox,
                'timeout': config.timeout,
                'rateLimit': 60000 / config.rate_limit,  # Convert to milliseconds
                'enableRateLimit': True,
                'verbose': False
            })
            
            # Initialize WebSocket exchange if available
            pro_exchange = None
            if hasattr(ccxtpro, config.name.lower()):
                pro_exchange_class = getattr(ccxtpro, config.name.lower())
                pro_exchange = pro_exchange_class({
                    'apiKey': config.api_key,
                    'secret': config.secret,
                    'sandbox': config.sandbox,
                    'timeout': config.timeout,
                    'verbose': False
                })
            
            # Test connection
            if self._test_connection(exchange):
                self.exchanges[config.name] = exchange
                if pro_exchange:
                    self.pro_exchanges[config.name] = pro_exchange
                
                self.configs[config.name] = config
                self.health_metrics[config.name] = ConnectionHealth(
                    exchange=config.name,
                    status=ExchangeStatus.CONNECTED,
                    last_ping=datetime.now()
                )
                self.rate_limiters[config.name] = {
                    'requests': 0,
                    'window_start': time.time(),
                    'limit': config.rate_limit
                }
                
                self.logger.info(f"Successfully added exchange: {config.name}")
                return True
            else:
                self.logger.error(f"Failed to connect to exchange: {config.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding exchange {config.name}: {str(e)}")
            return False
    
    def _test_connection(self, exchange: ccxt.Exchange) -> bool:
        """Test exchange connection"""
        try:
            exchange.load_markets()
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_ticker(self, symbol: str, exchange: Optional[str] = None) -> Optional[Dict]:
        """
        Get ticker data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            exchange: Specific exchange name (optional)
            
        Returns:
            Dict: Ticker data or None if failed
        """
        exchanges_to_try = [exchange] if exchange else list(self.exchanges.keys())
        
        for exchange_name in exchanges_to_try:
            if not self._check_rate_limit(exchange_name):
                continue
                
            try:
                exchange_obj = self.exchanges[exchange_name]
                ticker = exchange_obj.fetch_ticker(symbol)
                self._update_health_metrics(exchange_name, success=True)
                return {
                    'exchange': exchange_name,
                    'symbol': symbol,
                    'data': ticker,
                    'timestamp': datetime.now()
                }
            except Exception as e:
                self._update_health_metrics(exchange_name, success=False, error=str(e))
                self.logger.warning(f"Failed to get ticker from {exchange_name}: {str(e)}")
                continue
        
        return None
    
    def get_order_book(self, symbol: str, limit: int = 100, exchange: Optional[str] = None) -> Optional[Dict]:
        """
        Get order book data for a symbol
        
        Args:
            symbol: Trading symbol
            limit: Number of orders to fetch
            exchange: Specific exchange name (optional)
            
        Returns:
            Dict: Order book data or None if failed
        """
        exchanges_to_try = [exchange] if exchange else list(self.exchanges.keys())
        
        for exchange_name in exchanges_to_try:
            if not self._check_rate_limit(exchange_name):
                continue
                
            try:
                exchange_obj = self.exchanges[exchange_name]
                order_book = exchange_obj.fetch_order_book(symbol, limit)
                self._update_health_metrics(exchange_name, success=True)
                return {
                    'exchange': exchange_name,
                    'symbol': symbol,
                    'data': order_book,
                    'timestamp': datetime.now()
                }
            except Exception as e:
                self._update_health_metrics(exchange_name, success=False, error=str(e))
                self.logger.warning(f"Failed to get order book from {exchange_name}: {str(e)}")
                continue
        
        return None
    
    def get_trades(self, symbol: str, limit: int = 100, exchange: Optional[str] = None) -> Optional[Dict]:
        """
        Get recent trades for a symbol
        
        Args:
            symbol: Trading symbol
            limit: Number of trades to fetch
            exchange: Specific exchange name (optional)
            
        Returns:
            Dict: Trades data or None if failed
        """
        exchanges_to_try = [exchange] if exchange else list(self.exchanges.keys())
        
        for exchange_name in exchanges_to_try:
            if not self._check_rate_limit(exchange_name):
                continue
                
            try:
                exchange_obj = self.exchanges[exchange_name]
                trades = exchange_obj.fetch_trades(symbol, limit=limit)
                self._update_health_metrics(exchange_name, success=True)
                return {
                    'exchange': exchange_name,
                    'symbol': symbol,
                    'data': trades,
                    'timestamp': datetime.now()
                }
            except Exception as e:
                self._update_health_metrics(exchange_name, success=False, error=str(e))
                self.logger.warning(f"Failed to get trades from {exchange_name}: {str(e)}")
                continue
        
        return None
    
    async def subscribe_ticker(self, symbol: str, callback: Callable, exchange: Optional[str] = None):
        """
        Subscribe to real-time ticker updates via WebSocket
        
        Args:
            symbol: Trading symbol
            callback: Callback function for ticker updates
            exchange: Specific exchange name (optional)
        """
        exchanges_to_try = [exchange] if exchange else list(self.pro_exchanges.keys())
        
        for exchange_name in exchanges_to_try:
            try:
                exchange_obj = self.pro_exchanges[exchange_name]
                
                # Add callback
                if exchange_name not in self.callbacks:
                    self.callbacks[exchange_name] = []
                self.callbacks[exchange_name].append(callback)
                
                # Start WebSocket subscription
                while self.running:
                    try:
                        ticker = await exchange_obj.watch_ticker(symbol)
                        callback({
                            'exchange': exchange_name,
                            'symbol': symbol,
                            'data': ticker,
                            'timestamp': datetime.now()
                        })
                        self._update_health_metrics(exchange_name, success=True)
                    except Exception as e:
                        self._update_health_metrics(exchange_name, success=False, error=str(e))
                        self.logger.error(f"WebSocket error for {exchange_name}: {str(e)}")
                        await asyncio.sleep(5)  # Wait before reconnecting
                        
            except Exception as e:
                self.logger.error(f"Failed to subscribe to {exchange_name}: {str(e)}")
                continue
    
    def _check_rate_limit(self, exchange_name: str) -> bool:
        """Check if exchange is within rate limits"""
        if exchange_name not in self.rate_limiters:
            return False
        
        rate_limiter = self.rate_limiters[exchange_name]
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - rate_limiter['window_start'] >= 60:
            rate_limiter['requests'] = 0
            rate_limiter['window_start'] = current_time
        
        # Check if under limit
        if rate_limiter['requests'] >= rate_limiter['limit']:
            self.health_metrics[exchange_name].status = ExchangeStatus.RATE_LIMITED
            return False
        
        rate_limiter['requests'] += 1
        return True
    
    def _update_health_metrics(self, exchange_name: str, success: bool = True, error: str = None):
        """Update health metrics for an exchange"""
        if exchange_name not in self.health_metrics:
            return
        
        health = self.health_metrics[exchange_name]
        health.last_ping = datetime.now()
        
        if success:
            health.status = ExchangeStatus.CONNECTED
        else:
            health.error_count += 1
            health.last_error = error
            health.status = ExchangeStatus.ERROR
    
    def get_health_status(self) -> Dict[str, ConnectionHealth]:
        """Get health status of all exchanges"""
        return self.health_metrics.copy()
    
    def start_monitoring(self):
        """Start health monitoring thread"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_connections)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.info("Started exchange monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Stopped exchange monitoring")
    
    def _monitor_connections(self):
        """Monitor connection health"""
        while self.running:
            for exchange_name in self.exchanges.keys():
                try:
                    # Ping exchange
                    start_time = time.time()
                    self.exchanges[exchange_name].fetch_status()
                    latency = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Update health metrics
                    health = self.health_metrics[exchange_name]
                    health.latency = latency
                    health.last_ping = datetime.now()
                    
                    if health.status == ExchangeStatus.ERROR:
                        health.status = ExchangeStatus.CONNECTED
                        
                except Exception as e:
                    self._update_health_metrics(exchange_name, success=False, error=str(e))
            
            time.sleep(30)  # Check every 30 seconds
    
    def _load_config(self, config_file: str):
        """Load exchange configurations from file"""
        # Implementation for loading configurations from file
        # This would typically read from JSON, YAML, or other config format
        pass
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_monitoring()