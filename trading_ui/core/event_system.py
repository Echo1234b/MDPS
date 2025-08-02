from typing import Dict, List, Callable, Any

class EventSystem:
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}

    def register(self, event_type: str, handler: Callable):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def unregister(self, event_type: str, handler: Callable):
        if event_type in self.handlers:
            self.handlers[event_type].remove(handler)

    def emit(self, event_type: str, data: Any = None):
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(data)
