"""
Data Connectivity and Feed Integration Module

This module handles multi-exchange API integration, WebSocket streaming,
real-time data feeds, and connection management with automatic failover.
"""

from .exchange_api_manager import ExchangeAPIManager
from .mt5_connection import MT5Connection
from .bid_ask_streamer import BidAskStreamer
from .live_price_feed import LivePriceFeed
from .historical_data_loader import HistoricalDataLoader
from .ohlcv_extractor import OHLCVExtractor
from .order_book_snapshotter import OrderBookSnapshotter
from .tick_data_collector import TickDataCollector
from .volatility_index_tracker import VolatilityIndexTracker
from .volume_feed_integrator import VolumeFeedIntegrator
from .real_time_streaming_pipeline import RealTimeStreamingPipeline

__all__ = [
    'ExchangeAPIManager',
    'MT5Connection',
    'BidAskStreamer',
    'LivePriceFeed',
    'HistoricalDataLoader',
    'OHLCVExtractor',
    'OrderBookSnapshotter',
    'TickDataCollector',
    'VolatilityIndexTracker',
    'VolumeFeedIntegrator',
    'RealTimeStreamingPipeline'
]

__version__ = "1.0.0"
__author__ = "MDPS Development Team"
__description__ = "Data connectivity and feed integration for multi-exchange real-time data processing"