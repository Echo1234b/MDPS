import pytest
import pandas as pd
from datetime import datetime
from core.market_data import MarketData

def test_market_data_initialization():
    """Test market data initialization"""
    market_data = MarketData()
    assert len(market_data.symbols) == 0
    assert len(market_data.order_books) == 0

def test_market_data_update():
    """Test market data update functionality"""
    market_data = MarketData()
    test_data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [1000]
    })
    
    market_data.update_symbol_data('TEST', test_data)
    assert 'TEST' in market_data.symbols
    assert len(market_data.get_symbol_data('TEST')) == 1
    
    test_orderbook = {'bids': [[100, 10]], 'asks': [[101, 5]]}
    market_data.update_order_book('TEST', test_orderbook)
    assert 'TEST' in market_data.order_books
    assert market_data.get_order_book('TEST') == test_orderbook
