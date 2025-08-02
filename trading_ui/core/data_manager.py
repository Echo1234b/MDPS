from typing import Dict, Any
import pandas as pd

class DataManager:
    def __init__(self):
        self.data_cache: Dict[str, Any] = {}
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def update_data(self, data_type: str, data: Any):
        self.data_cache[data_type] = data
        self.notify_subscribers(data_type, data)

    def notify_subscribers(self, data_type: str, data: Any):
        for callback in self.subscribers:
            callback(data_type, data)
