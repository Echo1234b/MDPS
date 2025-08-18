"""
Bid/Ask Streamer Module

Real-time bid/ask price processing with microsecond precision,
spread analysis, market depth visualization, and liquidity assessment.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
import threading
import time
from enum import Enum

class StreamStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    RECONNECTING = "reconnecting"

@dataclass
class BidAskTick:
    """Individual bid/ask tick data"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    exchange: str = ""
    spread: float = field(init=False)
    mid_price: float = field(init=False)
    
    def __post_init__(self):
        self.spread = self.ask - self.bid if self.ask > 0 and self.bid > 0 else 0.0
        self.mid_price = (self.bid + self.ask) / 2 if self.ask > 0 and self.bid > 0 else 0.0

@dataclass
class SpreadAnalysis:
    """Spread analysis metrics"""
    symbol: str
    current_spread: float
    avg_spread: float
    min_spread: float
    max_spread: float
    spread_volatility: float
    spread_percentile_95: float
    spread_percentile_5: float
    tick_count: int
    analysis_period: timedelta

@dataclass
class LiquidityMetrics:
    """Liquidity assessment metrics"""
    symbol: str
    bid_depth: float
    ask_depth: float
    total_depth: float
    depth_imbalance: float  # (bid_depth - ask_depth) / total_depth
    weighted_spread: float
    liquidity_score: float

class BidAskStreamer:
    """
    Real-time bid/ask price processing with advanced analytics
    """
    
    def __init__(self, buffer_size: int = 10000):
        """
        Initialize Bid/Ask Streamer
        
        Args:
            buffer_size: Maximum number of ticks to keep in memory
        """
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size
        self.status = StreamStatus.STOPPED
        
        # Data storage
        self.tick_buffers: Dict[str, deque] = {}
        self.current_prices: Dict[str, BidAskTick] = {}
        self.spread_history: Dict[str, deque] = {}
        
        # Callbacks
        self.tick_callbacks: List[Callable] = []
        self.spread_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Analytics settings
        self.spread_analysis_window = timedelta(minutes=5)
        self.spread_alert_threshold = 2.0  # Standard deviations
        self.liquidity_update_interval = 1.0  # seconds
        
        # Threading
        self.analytics_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'ticks_processed': 0,
            'ticks_per_second': 0,
            'last_tick_time': None,
            'processing_latency': 0.0,
            'error_count': 0
        }
    
    def add_tick_callback(self, callback: Callable[[BidAskTick], None]):
        """Add callback for tick updates"""
        self.tick_callbacks.append(callback)
    
    def add_spread_callback(self, callback: Callable[[SpreadAnalysis], None]):
        """Add callback for spread analysis updates"""
        self.spread_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def process_tick(self, symbol: str, bid: float, ask: float, 
                    bid_size: float = 0.0, ask_size: float = 0.0, 
                    exchange: str = "") -> bool:
        """
        Process a new bid/ask tick
        
        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            bid_size: Bid size/volume
            ask_size: Ask size/volume
            exchange: Exchange name
            
        Returns:
            bool: Success status
        """
        try:
            start_time = time.perf_counter()
            
            # Validate input
            if bid <= 0 or ask <= 0 or bid >= ask:
                self.logger.warning(f"Invalid bid/ask for {symbol}: bid={bid}, ask={ask}")
                return False
            
            # Create tick object
            tick = BidAskTick(
                timestamp=datetime.now(),
                symbol=symbol,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                exchange=exchange
            )
            
            with self.lock:
                # Initialize buffers if needed
                if symbol not in self.tick_buffers:
                    self.tick_buffers[symbol] = deque(maxlen=self.buffer_size)
                    self.spread_history[symbol] = deque(maxlen=self.buffer_size)
                
                # Store tick
                self.tick_buffers[symbol].append(tick)
                self.current_prices[symbol] = tick
                self.spread_history[symbol].append(tick.spread)
                
                # Update metrics
                self.metrics['ticks_processed'] += 1
                self.metrics['last_tick_time'] = tick.timestamp
                self.metrics['processing_latency'] = (time.perf_counter() - start_time) * 1000
                
                # Calculate ticks per second
                if len(self.tick_buffers[symbol]) >= 2:
                    recent_ticks = list(self.tick_buffers[symbol])[-100:]  # Last 100 ticks
                    if len(recent_ticks) > 1:
                        time_diff = (recent_ticks[-1].timestamp - recent_ticks[0].timestamp).total_seconds()
                        if time_diff > 0:
                            self.metrics['ticks_per_second'] = len(recent_ticks) / time_diff
            
            # Trigger callbacks
            self._trigger_tick_callbacks(tick)
            
            # Check for spread alerts
            self._check_spread_alerts(symbol, tick)
            
            return True
            
        except Exception as e:
            self.metrics['error_count'] += 1
            self.logger.error(f"Error processing tick for {symbol}: {str(e)}")
            return False
    
    def get_current_price(self, symbol: str) -> Optional[BidAskTick]:
        """Get current bid/ask price for symbol"""
        with self.lock:
            return self.current_prices.get(symbol)
    
    def get_spread_analysis(self, symbol: str, period: Optional[timedelta] = None) -> Optional[SpreadAnalysis]:
        """
        Get spread analysis for symbol
        
        Args:
            symbol: Trading symbol
            period: Analysis period (default: 5 minutes)
            
        Returns:
            SpreadAnalysis: Spread analysis or None if insufficient data
        """
        if period is None:
            period = self.spread_analysis_window
        
        with self.lock:
            if symbol not in self.tick_buffers:
                return None
            
            # Get ticks within the period
            cutoff_time = datetime.now() - period
            recent_ticks = [tick for tick in self.tick_buffers[symbol] 
                           if tick.timestamp >= cutoff_time]
            
            if len(recent_ticks) < 2:
                return None
            
            # Calculate spread statistics
            spreads = [tick.spread for tick in recent_ticks]
            
            return SpreadAnalysis(
                symbol=symbol,
                current_spread=recent_ticks[-1].spread,
                avg_spread=np.mean(spreads),
                min_spread=np.min(spreads),
                max_spread=np.max(spreads),
                spread_volatility=np.std(spreads),
                spread_percentile_95=np.percentile(spreads, 95),
                spread_percentile_5=np.percentile(spreads, 5),
                tick_count=len(recent_ticks),
                analysis_period=period
            )
    
    def get_liquidity_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """
        Get liquidity metrics for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            LiquidityMetrics: Liquidity metrics or None if no data
        """
        with self.lock:
            current_tick = self.current_prices.get(symbol)
            if not current_tick:
                return None
            
            # Calculate depth metrics
            bid_depth = current_tick.bid_size
            ask_depth = current_tick.ask_size
            total_depth = bid_depth + ask_depth
            
            if total_depth == 0:
                depth_imbalance = 0.0
            else:
                depth_imbalance = (bid_depth - ask_depth) / total_depth
            
            # Calculate weighted spread
            if total_depth > 0:
                weighted_spread = current_tick.spread * (1 / total_depth)
            else:
                weighted_spread = current_tick.spread
            
            # Calculate liquidity score (higher is better)
            if current_tick.spread > 0 and total_depth > 0:
                liquidity_score = total_depth / current_tick.spread
            else:
                liquidity_score = 0.0
            
            return LiquidityMetrics(
                symbol=symbol,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                total_depth=total_depth,
                depth_imbalance=depth_imbalance,
                weighted_spread=weighted_spread,
                liquidity_score=liquidity_score
            )
    
    def get_tick_history(self, symbol: str, count: Optional[int] = None) -> List[BidAskTick]:
        """
        Get tick history for symbol
        
        Args:
            symbol: Trading symbol
            count: Number of ticks to return (default: all)
            
        Returns:
            List[BidAskTick]: List of ticks
        """
        with self.lock:
            if symbol not in self.tick_buffers:
                return []
            
            ticks = list(self.tick_buffers[symbol])
            if count is not None:
                ticks = ticks[-count:]
            
            return ticks
    
    def get_spread_history(self, symbol: str, count: Optional[int] = None) -> List[float]:
        """
        Get spread history for symbol
        
        Args:
            symbol: Trading symbol
            count: Number of spreads to return (default: all)
            
        Returns:
            List[float]: List of spreads
        """
        with self.lock:
            if symbol not in self.spread_history:
                return []
            
            spreads = list(self.spread_history[symbol])
            if count is not None:
                spreads = spreads[-count:]
            
            return spreads
    
    def start_analytics(self):
        """Start analytics processing thread"""
        if self.analytics_thread and self.analytics_thread.is_alive():
            return
        
        self.running = True
        self.status = StreamStatus.RUNNING
        self.analytics_thread = threading.Thread(target=self._analytics_worker)
        self.analytics_thread.daemon = True
        self.analytics_thread.start()
        self.logger.info("Started bid/ask analytics")
    
    def stop_analytics(self):
        """Stop analytics processing"""
        self.running = False
        self.status = StreamStatus.STOPPED
        if self.analytics_thread:
            self.analytics_thread.join()
        self.logger.info("Stopped bid/ask analytics")
    
    def _analytics_worker(self):
        """Analytics processing worker thread"""
        while self.running:
            try:
                # Generate spread analysis for all symbols
                with self.lock:
                    symbols = list(self.tick_buffers.keys())
                
                for symbol in symbols:
                    spread_analysis = self.get_spread_analysis(symbol)
                    if spread_analysis:
                        self._trigger_spread_callbacks(spread_analysis)
                
                time.sleep(self.liquidity_update_interval)
                
            except Exception as e:
                self.logger.error(f"Analytics worker error: {str(e)}")
                time.sleep(1)
    
    def _trigger_tick_callbacks(self, tick: BidAskTick):
        """Trigger tick callbacks"""
        for callback in self.tick_callbacks:
            try:
                callback(tick)
            except Exception as e:
                self.logger.error(f"Tick callback error: {str(e)}")
    
    def _trigger_spread_callbacks(self, analysis: SpreadAnalysis):
        """Trigger spread analysis callbacks"""
        for callback in self.spread_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                self.logger.error(f"Spread callback error: {str(e)}")
    
    def _trigger_alert_callbacks(self, alert_type: str, data: Dict):
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                self.logger.error(f"Alert callback error: {str(e)}")
    
    def _check_spread_alerts(self, symbol: str, tick: BidAskTick):
        """Check for spread-related alerts"""
        try:
            spread_analysis = self.get_spread_analysis(symbol, timedelta(minutes=1))
            if not spread_analysis or spread_analysis.tick_count < 10:
                return
            
            # Check for spread spike
            z_score = (tick.spread - spread_analysis.avg_spread) / spread_analysis.spread_volatility
            if abs(z_score) > self.spread_alert_threshold:
                alert_data = {
                    'symbol': symbol,
                    'current_spread': tick.spread,
                    'average_spread': spread_analysis.avg_spread,
                    'z_score': z_score,
                    'timestamp': tick.timestamp
                }
                self._trigger_alert_callbacks('spread_spike', alert_data)
            
            # Check for extremely wide spread
            if tick.spread > spread_analysis.spread_percentile_95 * 2:
                alert_data = {
                    'symbol': symbol,
                    'current_spread': tick.spread,
                    'percentile_95': spread_analysis.spread_percentile_95,
                    'timestamp': tick.timestamp
                }
                self._trigger_alert_callbacks('wide_spread', alert_data)
                
        except Exception as e:
            self.logger.error(f"Error checking spread alerts for {symbol}: {str(e)}")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for all symbols"""
        with self.lock:
            stats = {}
            for symbol in self.tick_buffers.keys():
                spread_analysis = self.get_spread_analysis(symbol)
                liquidity_metrics = self.get_liquidity_metrics(symbol)
                current_tick = self.current_prices.get(symbol)
                
                stats[symbol] = {
                    'tick_count': len(self.tick_buffers[symbol]),
                    'current_bid': current_tick.bid if current_tick else None,
                    'current_ask': current_tick.ask if current_tick else None,
                    'current_spread': current_tick.spread if current_tick else None,
                    'avg_spread': spread_analysis.avg_spread if spread_analysis else None,
                    'spread_volatility': spread_analysis.spread_volatility if spread_analysis else None,
                    'liquidity_score': liquidity_metrics.liquidity_score if liquidity_metrics else None,
                    'last_update': current_tick.timestamp if current_tick else None
                }
            
            return stats
    
    def clear_data(self, symbol: Optional[str] = None):
        """
        Clear stored data
        
        Args:
            symbol: Symbol to clear (if None, clears all)
        """
        with self.lock:
            if symbol:
                if symbol in self.tick_buffers:
                    self.tick_buffers[symbol].clear()
                    self.spread_history[symbol].clear()
                    if symbol in self.current_prices:
                        del self.current_prices[symbol]
            else:
                self.tick_buffers.clear()
                self.spread_history.clear()
                self.current_prices.clear()
                self.metrics['ticks_processed'] = 0
                self.metrics['error_count'] = 0
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_analytics()