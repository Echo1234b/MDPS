import pytest
from core.data_manager import DataManager

def test_data_manager_initialization():
    """Test data manager initialization"""
    manager = DataManager()
    assert len(manager.data_cache) == 0
    assert len(manager.subscribers) == 0

def test_data_manager_subscription():
    """Test data manager subscription functionality"""
    manager = DataManager()
    callback_called = False
    
    def callback(data_type, data):
        nonlocal callback_called
        callback_called = True
    
    manager.subscribe(callback)
    assert len(manager.subscribers) == 1
    
    manager.update_data('test_type', 'test_data')
    assert callback_called
    assert manager.data_cache['test_type'] == 'test_data'
