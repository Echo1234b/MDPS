from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QLineEdit, QComboBox,
                           QTabWidget, QTableWidget, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from ..widgets.charts.price_chart import PriceChart
from ..widgets.panels.position_panel import PositionPanel
from ..widgets.panels.order_panel import OrderPanel
from ..widgets.panels.risk_panel import RiskPanel

class TradingView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_trading_tools()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Order Entry Tab
        self.create_order_entry_tab()

        # Strategy Management Tab
        self.create_strategy_tab()

        # Risk Management Tab
        self.create_risk_management_tab()

    def create_order_entry_tab(self):
        """Create order entry tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Order entry section
        order_entry = QHBoxLayout()
        
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText('Symbol')
        
        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(1, 1000000)
        
        self.order_type = QComboBox()
        self.order_type.addItems(['Market', 'Limit', 'Stop', 'Stop Limit'])
        
        self.buy_button = QPushButton('Buy')
        self.sell_button = QPushButton('Sell')
        
        order_entry.addWidget(QLabel('Symbol:'))
        order_entry.addWidget(self.symbol_input)
        order_entry.addWidget(QLabel('Quantity:'))
        order_entry.addWidget(self.quantity_input)
        order_entry.addWidget(QLabel('Type:'))
        order_entry.addWidget(self.order_type)
        order_entry.addWidget(self.buy_button)
        order_entry.addWidget(self.sell_button)
        
        layout.addLayout(order_entry)

        # Price inputs (for limit/stop orders)
        self.price_inputs = QHBoxLayout()
        self.price_input = QDoubleSpinBox()
        self.stop_input = QDoubleSpinBox()
        
        self.price_inputs.addWidget(QLabel('Price:'))
        self.price_inputs.addWidget(self.price_input)
        self.price_inputs.addWidget(QLabel('Stop:'))
        self.price_inputs.addWidget(self.stop_input)
        
        layout.addLayout(self.price_inputs)

        # Active orders and positions
        panels_layout = QHBoxLayout()
        
        self.position_panel = PositionPanel()
        self.order_panel = OrderPanel()
        
        panels_layout.addWidget(self.position_panel)
        panels_layout.addWidget(self.order_panel)
        
        layout.addLayout(panels_layout)

        self.tab_widget.addTab(tab, "Order Entry")

    def create_strategy_tab(self):
        """Create strategy management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Strategy selection
        strategy_layout = QHBoxLayout()
        
        self.strategy_selector = QComboBox()
        self.strategy_selector.addItems(['Strategy 1', 'Strategy 2', 'Strategy 3'])
        
        self.activate_strategy = QPushButton('Activate')
        self.deactivate_strategy = QPushButton('Deactivate')
        
        strategy_layout.addWidget(QLabel('Strategy:'))
        strategy_layout.addWidget(self.strategy_selector)
        strategy_layout.addWidget(self.activate_strategy)
        strategy_layout.addWidget(self.deactivate_strategy)
        
        layout.addLayout(strategy_layout)

        # Strategy parameters
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Description'])
        layout.addWidget(self.params_table)

        # Strategy performance
        self.performance_chart = PriceChart()
        layout.addWidget(self.performance_chart)

        self.tab_widget.addTab(tab, "Strategy Management")

    def create_risk_management_tab(self):
        """Create risk management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Risk parameters
        risk_layout = QHBoxLayout()
        
        self.max_position = QSpinBox()
        self.max_position.setRange(1, 1000000)
        
        self.stop_loss = QDoubleSpinBox()
        self.take_profit = QDoubleSpinBox()
        
        risk_layout.addWidget(QLabel('Max Position:'))
        risk_layout.addWidget(self.max_position)
        risk_layout.addWidget(QLabel('Stop Loss:'))
        risk_layout.addWidget(self.stop_loss)
        risk_layout.addWidget(QLabel('Take Profit:'))
        risk_layout.addWidget(self.take_profit)
        
        layout.addLayout(risk_layout)

        # Risk metrics
        self.risk_panel = RiskPanel()
        layout.addWidget(self.risk_panel)

        # Risk alerts
        self.risk_alerts = QTableWidget()
        self.risk_alerts.setColumnCount(3)
        self.risk_alerts.setHorizontalHeaderLabels(['Time', 'Level', 'Message'])
        layout.addWidget(self.risk_alerts)

        self.tab_widget.addTab(tab, "Risk Management")

    def setup_trading_tools(self):
        """Setup trading tool connections"""
        self.event_system.register('order_update', self.update_orders)
        self.event_system.register('position_update', self.update_positions)
        self.event_system.register('strategy_update', self.update_strategy)
        self.event_system.register('risk_update', self.update_risk)

        # Connect signals
        self.buy_button.clicked.connect(self.place_buy_order)
        self.sell_button.clicked.connect(self.place_sell_order)
        self.order_type.currentTextChanged.connect(self.toggle_price_inputs)
        self.activate_strategy.clicked.connect(self.activate_selected_strategy)
        self.deactivate_strategy.clicked.connect(self.deactivate_selected_strategy)

    def toggle_price_inputs(self, order_type):
        """Toggle price inputs based on order type"""
        price_enabled = order_type in ['Limit', 'Stop Limit']
        stop_enabled = order_type == 'Stop Limit'
        
        self.price_input.setEnabled(price_enabled)
        self.stop_input.setEnabled(stop_enabled)

    def place_buy_order(self):
        """Place buy order"""
        order_data = {
            'symbol': self.symbol_input.text(),
            'quantity': self.quantity_input.value(),
            'type': self.order_type.currentText(),
            'side': 'buy',
            'price': self.price_input.value() if self.price_input.isEnabled() else None,
            'stop': self.stop_input.value() if self.stop_input.isEnabled() else None
        }
        self.event_system.emit('place_order', order_data)

    def place_sell_order(self):
        """Place sell order"""
        order_data = {
            'symbol': self.symbol_input.text(),
            'quantity': self.quantity_input.value(),
            'type': self.order_type.currentText(),
            'side': 'sell',
            'price': self.price_input.value() if self.price_input.isEnabled() else None,
            'stop': self.stop_input.value() if self.stop_input.isEnabled() else None
        }
        self.event_system.emit('place_order', order_data)

    def activate_selected_strategy(self):
        """Activate selected strategy"""
        strategy = self.strategy_selector.currentText()
        self.event_system.emit('activate_strategy', {'strategy': strategy})

    def deactivate_selected_strategy(self):
        """Deactivate selected strategy"""
        strategy = self.strategy_selector.currentText()
        self.event_system.emit('deactivate_strategy', {'strategy': strategy})

    def update_orders(self, orders):
        """Update orders display"""
        self.order_panel.update_orders(orders)

    def update_positions(self, positions):
        """Update positions display"""
        self.position_panel.update_position(positions)

    def update_strategy(self, data):
        """Update strategy display"""
        self.params_table.setRowCount(len(data['parameters']))
        for i, param in enumerate(data['parameters']):
            self.params_table.setItem(i, 0, QTableWidgetItem(param['name']))
            self.params_table.setItem(i, 1, QTableWidgetItem(str(param['value'])))
            self.params_table.setItem(i, 2, QTableWidgetItem(param['description']))
        
        self.performance_chart.update_data(data['performance'])

    def update_risk(self, risk_data):
        """Update risk display"""
        self.risk_panel.update_risk_metrics(risk_data['metrics'])
        
        # Update risk alerts
        self.risk_alerts.setRowCount(len(risk_data['alerts']))
        for i, alert in enumerate(risk_data['alerts']):
            self.risk_alerts.setItem(i, 0, QTableWidgetItem(alert['time']))
            self.risk_alerts.setItem(i, 1, QTableWidgetItem(alert['level']))
            self.risk_alerts.setItem(i, 2, QTableWidgetItem(alert['message']))
