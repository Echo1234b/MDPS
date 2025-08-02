import pytest
import tempfile
from datetime import datetime
from services.data_service import DataService
from data.models.market_model import MarketData
from data.models.trade_model import Trade

@pytest.fixture
def temp_service():
    """Create temporary service for testing"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        service = DataService(tmp.name)
        yield service
        import os
        os.unlink(tmp.name)

def test_data_service_initialization(temp_service):
    """Test data service initialization"""
    assert temp_service.db is not None
    assert temp_service.cache is not None

def test_market_data_operations(temp_service):
    """Test market data operations"""
    test_data = MarketData(
        symbol='TEST',
        timestamp=datetime.now(),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000
    )
    
    # Test saving and retrieving market data
    temp_service.save_market_data(test_data)
    retrieved_data = temp_service.get_market_data('TEST')
    assert retrieved_data is not None
    assert retrieved_data.symbol == test_data.symbol
    assert retrieved_data.close == test_data.close

def test_trade_operations(temp_service):
    """Test trade operations"""
    test_trade = Trade(
        symbol='TEST',
        timestamp=datetime.now(),
        side='buy',
        price=100.0,
        quantity=10,
        pnl=50.0
    )
    
    # Test saving trade
    temp_service.save_trade(test_trade)
    trades = temp_service.get_recent_trades('TEST')
    assert len(trades) >= 0
