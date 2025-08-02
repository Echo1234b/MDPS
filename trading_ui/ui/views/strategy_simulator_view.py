from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QTabWidget, QLabel, QTableWidget, QPushButton,
                           QComboBox, QDateEdit, QDoubleSpinBox, QSpinBox)
from ..widgets.charts.price_chart import PriceChart

class StrategySimulatorView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_simulation_tools()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Backtest Tab
        self.create_backtest_tab()

        # Optimization Tab
        self.create_optimization_tab()

        # Results Tab
        self.create_results_tab()

    def create_backtest_tab(self):
        """Create backtesting tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Backtest controls
        control_layout = QHBoxLayout()
        
        self.strategy = QComboBox()
        self.strategy.addItems(['Strategy 1', 'Strategy 2', 'Strategy 3'])
        
        self.start_date = QDateEdit()
        self.end_date = QDateEdit()
        
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(1000, 1000000)
        self.initial_capital.setValue(10000)
        
        self.run_backtest = QPushButton('Run Backtest')
        
        control_layout.addWidget(QLabel('Strategy:'))
        control_layout.addWidget(self.strategy)
        control_layout.addWidget(QLabel('Start:'))
        control_layout.addWidget(self.start_date)
        control_layout.addWidget(QLabel('End:'))
        control_layout.addWidget(self.end_date)
        control_layout.addWidget(QLabel('Capital:'))
        control_layout.addWidget(self.initial_capital)
        control_layout.addWidget(self.run_backtest)
        
        layout.addLayout(control_layout)

        # Backtest results chart
        self.backtest_chart = PriceChart()
        layout.addWidget(self.backtest_chart)

        # Trade history table
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(6)
        self.trades_table.setHorizontalHeaderLabels(['Time', 'Type', 'Price', 'Size', 'P&L', 'Cumulative'])
        layout.addWidget(self.trades_table)

        self.tab_widget.addTab(tab, "Backtest")

    def create_optimization_tab(self):
        """Create optimization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Optimization controls
        control_layout = QHBoxLayout()
        
        self.param1 = QDoubleSpinBox()
        self.param2 = QDoubleSpinBox()
        self.param3 = QDoubleSpinBox()
        
        self.optimization_target = QComboBox()
        self.optimization_target.addItems(['Profit Factor', 'Sharpe Ratio', 'Max Drawdown'])
        
        self.run_optimization = QPushButton('Run Optimization')
        
        control_layout.addWidget(QLabel('Param 1:'))
        control_layout.addWidget(self.param1)
        control_layout.addWidget(QLabel('Param 2:'))
        control_layout.addWidget(self.param2)
        control_layout.addWidget(QLabel('Param 3:'))
        control_layout.addWidget(self.param3)
        control_layout.addWidget(QLabel('Target:'))
        control_layout.addWidget(self.optimization_target)
        control_layout.addWidget(self.run_optimization)
        
        layout.addLayout(control_layout)

        # Optimization results
        self.optimization_chart = PriceChart()
        layout.addWidget(self.optimization_chart)

        # Parameter results table
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(5)
        self.params_table.setHorizontalHeaderLabels(['Param 1', 'Param 2', 'Param 3', 'Target', 'Rank'])
        layout.addWidget(self.params_table)

        self.tab_widget.addTab(tab, "Optimization")

    def create_results_tab(self):
        """Create results analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Performance metrics
        metrics_layout = QHBoxLayout()
        
        self.total_return = QLabel("Total Return: 0%")
        self.sharpe_ratio = QLabel("Sharpe Ratio: 0")
        self.max_drawdown = QLabel("Max Drawdown: 0%")
        self.win_rate = QLabel("Win Rate: 0%")
        
        metrics_layout.addWidget(self.total_return)
        metrics_layout.addWidget(self.sharpe_ratio)
        metrics_layout.addWidget(self.max_drawdown)
        metrics_layout.addWidget(self.win_rate)
        
        layout.addLayout(metrics_layout)

        # Equity curve
        self.equity_chart = PriceChart()
        layout.addWidget(self.equity_chart)

        # Detailed metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        layout.addWidget(self.metrics_table)

        self.tab_widget.addTab(tab, "Results")

    def setup_simulation_tools(self):
        """Setup simulation tool connections"""
        self.event_system.register('backtest_update', self.update_backtest)
        self.event_system.register('optimization_update', self.update_optimization)
        self.event_system.register('results_update', self.update_results)

        # Connect signals
        self.run_backtest.clicked.connect(self.run_strategy_backtest)
        self.run_optimization.clicked.connect(self.run_parameter_optimization)

    def run_strategy_backtest(self):
        """Run strategy backtest"""
        params = {
            'strategy': self.strategy.currentText(),
            'start_date': self.start_date.date().toString(),
            'end_date': self.end_date.date().toString(),
            'initial_capital': self.initial_capital.value()
        }
        self.event_system.emit('run_backtest', params)

    def run_parameter_optimization(self):
        """Run parameter optimization"""
        params = {
            'param1': self.param1.value(),
            'param2': self.param2.value(),
            'param3': self.param3.value(),
            'target': self.optimization_target.currentText()
        }
        self.event_system.emit('run_optimization', params)

    def update_backtest(self, data):
        """Update backtest results display"""
        self.backtest_chart.update_data(data['chart_data'])
        
        self.trades_table.setRowCount(len(data['trades']))
        for i, trade in enumerate(data['trades']):
            self.trades_table.setItem(i, 0, QTableWidgetItem(trade['time']))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade['type']))
            self.trades_table.setItem(i, 2, QTableWidgetItem(str(trade['price'])))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade['size'])))
            self.trades_table.setItem(i, 4, QTableWidgetItem(str(trade['pnl'])))
            self.trades_table.setItem(i, 5, QTableWidgetItem(str(trade['cumulative'])))

    def update_optimization(self, data):
        """Update optimization results display"""
        self.optimization_chart.update_data(data['chart_data'])
        
        self.params_table.setRowCount(len(data['params']))
        for i, param in enumerate(data['params']):
            self.params_table.setItem(i, 0, QTableWidgetItem(str(param['param1'])))
            self.params_table.setItem(i, 1, QTableWidgetItem(str(param['param2'])))
            self.params_table.setItem(i, 2, QTableWidgetItem(str(param['param3'])))
            self.params_table.setItem(i, 3, QTableWidgetItem(str(param['target'])))
            self.params_table.setItem(i, 4, QTableWidgetItem(str(param['rank'])))

    def update_results(self, data):
        """Update results analysis display"""
        # Update performance labels
        self.total_return.setText(f"Total Return: {data['total_return']}%")
        self.sharpe_ratio.setText(f"Sharpe Ratio: {data['sharpe_ratio']}")
        self.max_drawdown.setText(f"Max Drawdown: {data['max_drawdown']}%")
        self.win_rate.setText(f"Win Rate: {data['win_rate']}%")
        
        # Update equity curve
        self.equity_chart.update_data(data['equity_curve'])
        
        # Update metrics table
        self.metrics_table.setRowCount(len(data['metrics']))
        for i, metric in enumerate(data['metrics']):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric['name']))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(metric['value'])))
