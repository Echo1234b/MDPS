from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class BaseView(QWidget):
    def __init__(self, event_system, parent=None):
        super().__init__(parent)
        self.event_system = event_system
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI - to be overridden by subclasses"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
