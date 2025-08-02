from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QTabWidget, QLabel, QTableWidget, QPushButton,
                           QComboBox, QDoubleSpinBox, QSpinBox, QProgressBar)
from ..widgets.charts.price_chart import PriceChart

class RiskManagementView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_risk_tools()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Position Risk Tab
        self.create_position_risk_tab()

        # Portfolio Risk Tab
        self.create_portfolio_risk_tab()

        # Risk Alerts Tab
        self.create_risk_alerts_tab()

    def create_position_risk_tab(self):
        """Create position risk analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Risk controls
        control_layout = QHBoxLayout()
        
        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(0, 1000000)
        
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(0, 100)
        
        self.take_profit = QDoubleSpinBox()
        self.take_profit.setRange(0, 100)
        
        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0, 100)
        self.risk_per_trade.setValue(2)
        
        control_layout.addWidget(QLabel('Position Size:'))
        control_layout.addWidget(self.position_size)
        control_layout.addWidget(QLabel('Stop Loss %:'))
        control_layout.addWidget(self.stop_loss)
        control_layout.addWidget(QLabel('Take Profit %:'))
        control_layout.addWidget(self.take_profit)
        control_layout.addWidget(QLabel('Risk %:'))
        control_layout.addWidget(self.risk_per_trade)
        
        layout.addLayout(control_layout)

        # Risk metrics
        metrics_layout = QHBoxLayout()
        
        self.current_risk = self.create_risk_metric("Current Risk")
        self.max_risk = self.create_risk_metric("Max Risk")
        self.risk_reward = self.create_risk_metric("Risk/Reward")
        
        metrics_layout.addWidget(self.current_risk)
        metrics_layout.addWidget(self.max_risk)
        metrics_layout.addWidget(self.risk_reward)
        
        layout.addLayout(metrics_layout)

        # Position risk chart
        self.position_chart = PriceChart()
        layout.addWidget(self.position_chart)

        # Risk details table
        self.risk_table = QTableWidget()
        self.risk_table.setColumnCount(5)
        self.risk_table.setHorizontalHeaderLabels(['Position', 'Size', 'Entry', 'Stop Loss', 'Risk'])
        layout.addWidget(self.risk_table)

        self.tab_widget.addTab(tab, "Position Risk")

    def create_portfolio_risk_tab(self):
        """Create portfolio risk analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Portfolio controls
        control_layout = QHBoxLayout()
        
        self.max_portfolio_risk = QDoubleSpinBox()
        self.max_portfolio_risk.setRange(0, 100)
        self.max_portfolio_risk.setValue(10)
        
        self.correlation_threshold = QDoubleSpinBox()
        self.correlation_threshold.setRange(0, 1)
        self.correlation_threshold.setValue(0.7)
        
        self.rebalance_button = QPushButton('Rebalance')
        
        control_layout.addWidget(QLabel('Max Risk %:'))
        control_layout.addWidget(self.max_portfolio_risk)
        control_layout.addWidget(QLabel('Correlation:'))
        control_layout.addWidget(self.correlation_threshold)
        control_layout.addWidget(self.rebalance_button)
        
        layout.addLayout(control_layout)

        # Portfolio metrics
        portfolio_layout = QHBoxLayout()
        
        self.total_exposure = self.create_risk_metric("Total Exposure")
        self.var = self.create_risk_metric("Value at Risk")
        self.sharpe = self.create_risk_metric("Sharpe Ratio")
        
        portfolio_layout.addWidget(self.total_exposure)
        portfolio_layout.addWidget(self.var)
        portfolio_layout.addWidget(self.sharpe)
        
        layout.addLayout(portfolio_layout)

        # Portfolio risk chart
        self.portfolio_chart = PriceChart()
        layout.addWidget(self.portfolio_chart)

        # Correlation matrix
        self.correlation_table = QTableWidget()
        layout.addWidget(self.correlation_table)

        self.tab_widget.addTab(tab, "Portfolio Risk")

    def create_risk_alerts_tab(self):
        """Create risk alerts tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Alert controls
        control_layout = QHBoxLayout()
        
        self.alert_type = QComboBox()
        self.alert_type.addItems(['All', 'Position Risk', 'Portfolio Risk', 'Margin'])
        
        self.severity = QComboBox()
        self.severity.addItems(['All', 'High', 'Medium', 'Low'])
        
        self.clear_alerts = QPushButton('Clear Alerts')
        
        control_layout.addWidget(QLabel('Type:'))
        control_layout.addWidget(self.alert_type)
        control_layout.addWidget(QLabel('Severity:'))
        control_layout.addWidget(self.severity)
        control_layout.addWidget(self.clear_alerts)
        
        layout.addLayout(control_layout)

        # Active alerts
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(4)
        self.alerts_table.setHorizontalHeaderLabels(['Time', 'Type', 'Severity', 'Message'])
        layout.addWidget(self.alerts_table)

        # Risk history chart
        self.risk_history_chart = PriceChart()
        layout.addWidget(self.risk_history_chart)

        self.tab_widget.addTab(tab, "Risk Alerts")

    def create_risk_metric(self, label):
        """Create risk metric widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label_widget = QLabel(label)
        value_label = QLabel("0")
        progress = QProgressBar()
        progress.setRange(0, 100)
        
        layout.addWidget(label_widget)
        layout.addWidget(value_label)
        layout.addWidget(progress)
        
        return widget

    def setup_risk_tools(self):
        """Setup risk management connections"""
        self.event_system.register('position_risk_update', self.update_position_risk)
        self.event_system.register('portfolio_risk_update', self.update_portfolio_risk)
        self.event_system.register('risk_alert', self.update_risk_alerts)

        # Connect signals
        self.rebalance_button.clicked.connect(self.rebalance_portfolio)
        self.clear_alerts.clicked.connect(self.clear_risk_alerts)

    def rebalance_portfolio(self):
        """Rebalance portfolio based on risk parameters"""
        params = {
            'max_risk': self.max_portfolio_risk.value(),
            'correlation': self.correlation_threshold.value()
        }
        self.event_system.emit('rebalance_portfolio', params)

    def clear_risk_alerts(self):
        """Clear risk alerts"""
        self.event_system.emit('clear_risk_alerts', {
            'type': self.alert_type.currentText(),
            'severity': self.severity.currentText()
        })

    def update_position_risk(self, data):
        """Update position risk display"""
        # Update risk metrics
        current_risk_value = self.current_risk.findChildren(QLabel)[1]
        current_risk_progress = self.current_risk.findChildren(QProgressBar)[0]
        current_risk_value.setText(f"{data['current_risk']:.2f}%")
        current_risk_progress.setValue(int(data['current_risk']))
        
        max_risk_value = self.max_risk.findChildren(QLabel)[1]
        max_risk_progress = self.max_risk.findChildren(QProgressBar)[0]
        max_risk_value.setText(f"{data['max_risk']:.2f}%")
        max_risk_progress.setValue(int(data['max_risk']))
        
        risk_reward_value = self.risk_reward.findChildren(QLabel)[1]
        risk_reward_value.setText(f"{data['risk_reward']:.2f}")
        
        # Update position chart
        self.position_chart.update_data(data['chart_data'])
        
        # Update risk table
        self.risk_table.setRowCount(len(data['positions']))
        for i, pos in enumerate(data['positions']):
            self.risk_table.setItem(i, 0, QTableWidgetItem(pos['symbol']))
            self.risk_table.setItem(i, 1, QTableWidgetItem(str(pos['size'])))
            self.risk_table.setItem(i, 2, QTableWidgetItem(str(pos['entry'])))
            self.risk_table.setItem(i, 3, QTableWidgetItem(str(pos['stop_loss'])))
            self.risk_table.setItem(i, 4, QTableWidgetItem(str(pos['risk'])))

    def update_portfolio_risk(self, data):
        """Update portfolio risk display"""
        # Update portfolio metrics
        exposure_value = self.total_exposure.findChildren(QLabel)[1]
        exposure_progress = self.total_exposure.findChildren(QProgressBar)[0]
        exposure_value.setText(f"{data['exposure']:.2f}%")
        exposure_progress.setValue(int(data['exposure']))
        
        var_value = self.var.findChildren(QLabel)[1]
        var_value.setText(f"{data['var']:.2f}")
        
        sharpe_value = self.sharpe.findChildren(QLabel)[1]
        sharpe_value.setText(f"{data['sharpe']:.2f}")
        
        # Update portfolio chart
        self.portfolio_chart.update_data(data['chart_data'])
        
        # Update correlation matrix
        self.correlation_table.setRowCount(len(data['correlation_matrix']))
        self.correlation_table.setColumnCount(len(data['correlation_matrix'][0]))
        for i, row in enumerate(data['correlation_matrix']):
            for j, value in enumerate(row):
                self.correlation_table.setItem(i, j, QTableWidgetItem(f"{value:.3f}"))

    def update_risk_alerts(self, data):
        """Update risk alerts display"""
        self.alerts_table.setRowCount(len(data['alerts']))
        for i, alert in enumerate(data['alerts']):
            self.alerts_table.setItem(i, 0, QTableWidgetItem(alert['time']))
            self.alerts_table.setItem(i, 1, QTableWidgetItem(alert['type']))
            self.alerts_table.setItem(i, 2, QTableWidgetItem(alert['severity']))
            self.alerts_table.setItem(i, 3, QTableWidgetItem(alert['message']))
        
        # Update risk history chart
        self.risk_history_chart.update_data(data['history'])
