from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                           QSplitter, QLabel, QProgressBar)
from PyQt5.QtCore import Qt
from ..widgets.charts.price_chart import PriceChart
from ..widgets.charts.orderbook_chart import OrderBookChart
from ..widgets.charts.volume_profile import VolumeProfile
from ..widgets.tables.market_table import MarketTable

class MarketView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_data_monitors()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Add feed status section
        self.create_feed_status_section(layout)

        # Create splitter for flexible layout
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left side - Price chart and volume profile
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.price_chart = PriceChart()
        self.volume_profile = VolumeProfile()
        
        left_layout.addWidget(self.price_chart)
        left_layout.addWidget(self.volume_profile)

        # Right side - Order book and market table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.orderbook_chart = OrderBookChart()
        self.market_table = MarketTable()
        
        right_layout.addWidget(self.orderbook_chart)
        right_layout.addWidget(self.market_table)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 500])

    def create_feed_status_section(self, layout):
        """Create feed status monitoring section"""
        status_layout = QHBoxLayout()
        
        self.mt5_status = QLabel("MT5: Disconnected")
        self.api_status = QLabel("API: Disconnected")
        self.latency_bar = QProgressBar()
        self.latency_bar.setRange(0, 1000)
        
        status_layout.addWidget(self.mt5_status)
        status_layout.addWidget(self.api_status)
        status_layout.addWidget(QLabel("Latency:"))
        status_layout.addWidget(self.latency_bar)
        
        layout.addLayout(status_layout)

    def setup_data_monitors(self):
        """Setup data monitoring connections"""
        self.event_system.register('feed_status_update', self.update_feed_status)
        self.event_system.register('market_data_update', self.update_market_data)

    def update_feed_status(self, status_data):
        """Update feed status indicators"""
        self.mt5_status.setText(f"MT5: {status_data['mt5_status']}")
        self.api_status.setText(f"API: {status_data['api_status']}")
        self.latency_bar.setValue(int(status_data['latency']))

    def update_market_data(self, data):
        """Update market data displays"""
        self.price_chart.update_data(data['price_data'])
        self.orderbook_chart.update_orderbook(data['bids'], data['asks'])
        self.volume_profile.update_volume_profile(data['prices'], data['volumes'])
        self.market_table.update_market_data(data['market_summary'])
