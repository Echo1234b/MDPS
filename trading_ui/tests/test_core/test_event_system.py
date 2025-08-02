import pytest
from core.event_system import EventSystem

def test_event_system_initialization():
    """Test event system initialization"""
    event_system = EventSystem()
    assert len(event_system.handlers) == 0

def test_event_system_subscription():
    """Test event system subscription functionality"""
    event_system = EventSystem()
    callback_called = False
    
    def handler(data):
        nonlocal callback_called
        callback_called = True
    
    event_system.register('test_event', handler)
    assert 'test_event' in event_system.handlers
    assert len(event_system.handlers['test_event']) == 1
    
    event_system.emit('test_event', 'test_data')
    assert callback_called
    
    event_system.unregister('test_event', handler)
    assert len(event_system.handlers['test_event']) == 0
