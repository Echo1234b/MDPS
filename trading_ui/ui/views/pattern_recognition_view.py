from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QTabWidget, QLabel, QTableWidget, QPushButton,
                           QComboBox, QGraphicsView, QGraphicsScene)
from ..widgets.charts.price_chart import PriceChart

class PatternRecognitionView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_pattern_analysis()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Elliott Waves & Harmonics Tab
        self.create_elliott_harmonics_tab()

        # Fibonacci & Geometry Tab
        self.create_fibonacci_tab()

        # Price Action Context Tab
        self.create_price_action_tab()

    def create_elliott_harmonics_tab(self):
        """Create Elliott waves and harmonics analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Analysis controls
        control_layout = QHBoxLayout()
        
        self.wave_type = QComboBox()
        self.wave_type.addItems(['Impulse', 'Correction', 'All'])
        
        self.pattern_type = QComboBox()
        self.pattern_type.addItems(['Gartley', 'Bat', 'Crab', 'All'])
        
        self.analyze_patterns = QPushButton('Analyze')
        
        control_layout.addWidget(QLabel('Wave Type:'))
        control_layout.addWidget(self.wave_type)
        control_layout.addWidget(QLabel('Pattern Type:'))
        control_layout.addWidget(self.pattern_type)
        control_layout.addWidget(self.analyze_patterns)
        
        layout.addLayout(control_layout)

        # Pattern visualization
        self.pattern_chart = PriceChart()
        layout.addWidget(self.pattern_chart)

        # Pattern details table
        self.patterns_table = QTableWidget()
        self.patterns_table.setColumnCount(5)
        self.patterns_table.setHorizontalHeaderLabels(['Type', 'Start', 'End', 'Strength', 'Status'])
        layout.addWidget(self.patterns_table)

        self.tab_widget.addTab(tab, "Elliott & Harmonics")

    def create_fibonacci_tab(self):
        """Create Fibonacci and geometry analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Fibonacci controls
        control_layout = QHBoxLayout()
        
        self.fib_type = QComboBox()
        self.fib_type.addItems(['Retracement', 'Extension', 'Fan', 'Arc'])
        
        self.draw_fib = QPushButton('Draw')
        
        control_layout.addWidget(QLabel('Type:'))
        control_layout.addWidget(self.fib_type)
        control_layout.addWidget(self.draw_fib)
        
        layout.addLayout(control_layout)

        # Fibonacci visualization
        self.fib_view = QGraphicsView()
        self.fib_scene = QGraphicsScene()
        self.fib_view.setScene(self.fib_scene)
        layout.addWidget(self.fib_view)

        # Fibonacci levels table
        self.fib_table = QTableWidget()
        self.fib_table.setColumnCount(3)
        self.fib_table.setHorizontalHeaderLabels(['Level', 'Price', 'Strength'])
        layout.addWidget(self.fib_table)

        self.tab_widget.addTab(tab, "Fibonacci & Geometry")

    def create_price_action_tab(self):
        """Create price action analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Price action controls
        control_layout = QHBoxLayout()
        
        self.action_type = QComboBox()
        self.action_type.addItems(['SuperTrend', 'Ichimoku', 'Price Action'])
        
        self.analyze_action = QPushButton('Analyze')
        
        control_layout.addWidget(QLabel('Type:'))
        control_layout.addWidget(self.action_type)
        control_layout.addWidget(self.analyze_action)
        
        layout.addLayout(control_layout)

        # Price action visualization
        self.action_chart = PriceChart()
        layout.addWidget(self.action_chart)

        # Price action signals table
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(4)
        self.signals_table.setHorizontalHeaderLabels(['Time', 'Signal', 'Strength', 'Type'])
        layout.addWidget(self.signals_table)

        self.tab_widget.addTab(tab, "Price Action")

    def setup_pattern_analysis(self):
        """Setup pattern analysis connections"""
        self.event_system.register('pattern_update', self.update_patterns)
        self.event_system.register('fibonacci_update', self.update_fibonacci)
        self.event_system.register('price_action_update', self.update_price_action)

        # Connect signals
        self.analyze_patterns.clicked.connect(self.analyze_wave_patterns)
        self.draw_fib.clicked.connect(self.draw_fibonacci_levels)
        self.analyze_action.clicked.connect(self.analyze_price_action)

    def analyze_wave_patterns(self):
        """Analyze wave patterns"""
        params = {
            'wave_type': self.wave_type.currentText(),
            'pattern_type': self.pattern_type.currentText()
        }
        self.event_system.emit('analyze_patterns', params)

    def draw_fibonacci_levels(self):
        """Draw Fibonacci levels"""
        params = {
            'fib_type': self.fib_type.currentText()
        }
        self.event_system.emit('draw_fibonacci', params)

    def analyze_price_action(self):
        """Analyze price action"""
        params = {
            'action_type': self.action_type.currentText()
        }
        self.event_system.emit('analyze_price_action', params)

    def update_patterns(self, data):
        """Update patterns display"""
        self.pattern_chart.update_data(data['chart_data'])
        
        self.patterns_table.setRowCount(len(data['patterns']))
        for i, pattern in enumerate(data['patterns']):
            self.patterns_table.setItem(i, 0, QTableWidgetItem(pattern['type']))
            self.patterns_table.setItem(i, 1, QTableWidgetItem(pattern['start']))
            self.patterns_table.setItem(i, 2, QTableWidgetItem(pattern['end']))
            self.patterns_table.setItem(i, 3, QTableWidgetItem(str(pattern['strength'])))
            self.patterns_table.setItem(i, 4, QTableWidgetItem(pattern['status']))

    def update_fibonacci(self, data):
        """Update Fibonacci display"""
        self.fib_scene.clear()
        
        # Draw Fibonacci levels
        for level in data['levels']:
            self.fib_scene.addLine(
                level['x1'], level['y1'],
                level['x2'], level['y2']
            )
        
        # Update Fibonacci table
        self.fib_table.setRowCount(len(data['levels']))
        for i, level in enumerate(data['levels']):
            self.fib_table.setItem(i, 0, QTableWidgetItem(str(level['level'])))
            self.fib_table.setItem(i, 1, QTableWidgetItem(str(level['price'])))
            self.fib_table.setItem(i, 2, QTableWidgetItem(str(level['strength'])))

    def update_price_action(self, data):
        """Update price action display"""
        self.action_chart.update_data(data['chart_data'])
        
        self.signals_table.setRowCount(len(data['signals']))
        for i, signal in enumerate(data['signals']):
            self.signals_table.setItem(i, 0, QTableWidgetItem(signal['time']))
            self.signals_table.setItem(i, 1, QTableWidgetItem(signal['signal']))
            self.signals_table.setItem(i, 2, QTableWidgetItem(str(signal['strength'])))
            self.signals_table.setItem(i, 3, QTableWidgetItem(signal['type']))
