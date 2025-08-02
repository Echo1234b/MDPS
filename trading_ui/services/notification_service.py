from typing import Callable, List
from ..core.event_system import EventSystem

class NotificationService:
    def __init__(self, event_system: EventSystem):
        self.event_system = event_system
        self.callbacks = []
        self.setup_event_handlers()

    def setup_event_handlers(self):
        """Setup event handlers for notifications"""
        self.event_system.register('order_placed', self.on_order_placed)
        self.event_system.register('order_cancelled', self.on_order_cancelled)
        self.event_system.register('position_updated', self.on_position_updated)

    def register_callback(self, callback: Callable):
        """Register callback for notifications"""
        self.callbacks.append(callback)

    def notify(self, message: str, level: str = 'info'):
        """Send notification to all registered callbacks"""
        for callback in self.callbacks:
            callback(message, level)

    def on_order_placed(self, order):
        """Handle order placed event"""
        self.notify(f"Order placed: {order['symbol']} {order['side']} {order['quantity']}")

    def on_order_cancelled(self, order):
        """Handle order cancelled event"""
        self.notify(f"Order cancelled: {order['id']}")

    def on_position_updated(self, position):
        """Handle position updated event"""
        self.notify(f"Position updated: {position['symbol']} {position['quantity']}")
