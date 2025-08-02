from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QTabWidget, QLabel, QTableWidget, QPushButton,
                           QComboBox, QGraphicsView, QGraphicsScene)
from ..widgets.charts.price_chart import PriceChart

class MarketStructureView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_structure_analysis()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Key Market Zones Tab
        self.create_market_zones_tab()

        # Volume/Liquidity Structure Tab
        self.create_liquidity_tab()

        # Market Regime & Structure Tab
        self.create_regime_tab()

    def create_market_zones_tab(self):
        """Create market zones analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Zone analysis controls
        control_layout = QHBoxLayout()
        
        self.zone_type = QComboBox()
        self.zone_type.addItems(['Support/Resistance', 'Supply/Demand', 'POI'])
        
        self.timeframe = QComboBox()
        self.timeframe.addItems(['1m', '5m', '15m', '1h', '4h', '1d'])
        
        self.analyze_zones = QPushButton('Analyze')
        
        control_layout.addWidget(QLabel('Zone Type:'))
        control_layout.addWidget(self.zone_type)
        control_layout.addWidget(QLabel('Timeframe:'))
        control_layout.addWidget(self.timeframe)
        control_layout.addWidget(self.analyze_zones)
        
        layout.addLayout(control_layout)

        # Price chart with zones
        self.zone_chart = PriceChart()
        layout.addWidget(self.zone_chart)

        # Zones table
        self.zones_table = QTableWidget()
        self.zones_table.setColumnCount(4)
        self.zones_table.setHorizontalHeaderLabels(['Level', 'Type', 'Strength', 'Tests'])
        layout.addWidget(self.zones_table)

        self.tab_widget.addTab(tab, "Market Zones")

    def create_liquidity_tab(self):
        """Create liquidity structure analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Liquidity controls
        control_layout = QHBoxLayout()
        
        self.liquidity_type = QComboBox()
        self.liquidity_type.addItems(['VWAP', 'Volume Profile', 'FVG', 'Liquidity Gaps'])
        
        self.refresh_liquidity = QPushButton('Refresh')
        
        control_layout.addWidget(QLabel('Type:'))
        control_layout.addWidget(self.liquidity_type)
        control_layout.addWidget(self.refresh_liquidity)
        
        layout.addLayout(control_layout)

        # Liquidity visualization
        self.liquidity_view = QGraphicsView()
        self.liquidity_scene = QGraphicsScene()
        self.liquidity_view.setScene(self.liquidity_scene)
        layout.addWidget(self.liquidity_view)

        # Liquidity metrics table
        self.liquidity_table = QTableWidget()
        self.liquidity_table.setColumnCount(3)
        self.liquidity_table.setHorizontalHeaderLabels(['Level', 'Volume', 'Type'])
        layout.addWidget(self.liquidity_table)

        self.tab_widget.addTab(tab, "Liquidity Structure")

    def create_regime_tab(self):
        """Create market regime analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Regime analysis controls
        control_layout = QHBoxLayout()
        
        self.regime_type = QComboBox()
        self.regime_type.addItems(['Trend', 'Range', 'Breakout', 'Reversal'])
        
        self.detect_regime = QPushButton('Detect')
        
        control_layout.addWidget(QLabel('Regime Type:'))
        control_layout.addWidget(self.regime_type)
        control_layout.addWidget(self.detect_regime)
        
        layout.addLayout(control_layout)

        # Market structure visualization
        self.structure_chart = PriceChart()
        layout.addWidget(self.structure_chart)

        # Structure events table
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(4)
        self.events_table.setHorizontalHeaderLabels(['Time', 'Event', 'Type', 'Strength'])
        layout.addWidget(self.events_table)

        self.tab_widget.addTab(tab, "Market Regime")

    def setup_structure_analysis(self):
        """Setup structure analysis connections"""
        self.event_system.register('market_zones_update', self.update_zones)
        self.event_system.register('liquidity_update', self.update_liquidity)
        self.event_system.register('regime_update', self.update_regime)

        # Connect signals
        self.analyze_zones.clicked.connect(self.analyze_market_zones)
        self.refresh_liquidity.clicked.connect(self.refresh_liquidity_data)
        self.detect_regime.clicked.connect(self.detect_market_regime)

    def analyze_market_zones(self):
        """Analyze market zones"""
        params = {
            'zone_type': self.zone_type.currentText(),
            'timeframe': self.timeframe.currentText()
        }
        self.event_system.emit('analyze_zones', params)

    def refresh_liquidity_data(self):
        """Refresh liquidity data"""
        params = {
            'liquidity_type': self.liquidity_type.currentText()
        }
        self.event_system.emit('refresh_liquidity', params)

    def detect_market_regime(self):
        """Detect market regime"""
        params = {
            'regime_type': self.regime_type.currentText()
        }
        self.event_system.emit('detect_regime', params)

    def update_zones(self, data):
        """Update market zones display"""
        self.zone_chart.update_data(data['chart_data'])
        
        self.zones_table.setRowCount(len(data['zones']))
        for i, zone in enumerate(data['zones']):
            self.zones_table.setItem(i, 0, QTableWidgetItem(str(zone['level'])))
            self.zones_table.setItem(i, 1, QTableWidgetItem(zone['type']))
            self.zones_table.setItem(i, 2, QTableWidgetItem(str(zone['strength'])))
            self.zones_table.setItem(i, 3, QTableWidgetItem(str(zone['tests'])))

    def update_liquidity(self, data):
        """Update liquidity structure display"""
        self.liquidity_scene.clear()
        
        # Draw liquidity visualization
        for level in data['levels']:
            self.liquidity_scene.addRect(
                level['x'], level['y'],
                level['width'], level['height']
            )
        
        # Update liquidity table
        self.liquidity_table.setRowCount(len(data['levels']))
        for i, level in enumerate(data['levels']):
            self.liquidity_table.setItem(i, 0, QTableWidgetItem(str(level['price'])))
            self.liquidity_table.setItem(i, 1, QTableWidgetItem(str(level['volume'])))
            self.liquidity_table.setItem(i, 2, QTableWidgetItem(level['type']))

    def update_regime(self, data):
        """Update market regime display"""
        self.structure_chart.update_data(data['chart_data'])
        
        self.events_table.setRowCount(len(data['events']))
        for i, event in enumerate(data['events']):
            self.events_table.setItem(i, 0, QTableWidgetItem(event['time']))
            self.events_table.setItem(i, 1, QTableWidgetItem(event['event']))
            self.events_table.setItem(i, 2, QTableWidgetItem(event['type']))
            self.events_table.setItem(i, 3, QTableWidgetItem(str(event['strength'])))
