import pytest
from datetime import datetime
from data.models.market_model import MarketData
from data.models.trade_model import Trade

def test_market_model():
    """Test market data model"""
    data = MarketData(
        symbol='TEST',
        timestamp=datetime.now(),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000
    )
    
    # Test to_dict conversion
    data_dict = data.to_dict()
    assert data_dict['symbol'] == 'TEST'
    assert 'timestamp' in data_dict
    
    # Test from_dict creation
    new_data = MarketData.from_dict(data_dict)
    assert new_data.symbol == data.symbol
    assert new_data.open == data.open
    assert new_data.high == data.high
    assert new_data.low == data.low
    assert new_data.close == data.close
    assert new_data.volume == data.volume

def test_trade_model():
    """Test trade model"""
    trade = Trade(
        symbol='TEST',
        timestamp=datetime.now(),
        side='buy',
        price=100.0,
        quantity=10,
        pnl=50.0
    )
    
    # Test to_dict conversion
    trade_dict = trade.to_dict()
    assert trade_dict['symbol'] == 'TEST'
    assert trade_dict['side'] == 'buy'
    
    # Test from_dict creation
    new_trade = Trade.from_dict(trade_dict)
    assert new_trade.symbol == trade.symbol
    assert new_trade.side == trade.side
    assert new_trade.price == trade.price
    assert new_trade.quantity == trade.quantity
    assert new_trade.pnl == trade.pnl
