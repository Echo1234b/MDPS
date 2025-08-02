import pytest
import sqlite3
import tempfile
from datetime import datetime
from data.database import Database

@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db = Database(tmp.name)
        yield db
        import os
        os.unlink(tmp.name)

def test_database_initialization(temp_db):
    """Test database initialization"""
    with sqlite3.connect(temp_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert 'market_data' in tables
        assert 'trades' in tables
        assert 'orders' in tables

def test_market_data_insertion(temp_db):
    """Test market data insertion"""
    test_data = {
        'symbol': 'TEST',
        'timestamp': datetime.now(),
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.5,
        'volume': 1000
    }
    
    temp_db.insert_market_data(test_data)
    
    with sqlite3.connect(temp_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM market_data WHERE symbol = 'TEST'")
        result = cursor.fetchone()
        assert result is not None
        assert result[1] == 'TEST'
