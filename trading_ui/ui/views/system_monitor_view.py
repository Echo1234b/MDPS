from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QProgressBar, QTabWidget, QTableWidget,
                           QPushButton, QComboBox, QTableWidgetItem)
from PyQt5.QtCore import Qt
from ..widgets.charts.price_chart import PriceChart

class SystemMonitorView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_monitoring()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # System Metrics Tab
        self.create_system_metrics_tab()

        # Alerts Tab
        self.create_alerts_tab()

        # Performance Tab
        self.create_performance_tab()

    def create_system_metrics_tab(self):
        """Create system metrics monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # System health indicators
        health_layout = QHBoxLayout()
        
        self.cpu_usage = self.create_metric_widget("CPU Usage")
        self.memory_usage = self.create_metric_widget("Memory Usage")
        self.disk_usage = self.create_metric_widget("Disk Usage")
        
        health_layout.addWidget(self.cpu_usage)
        health_layout.addWidget(self.memory_usage)
        health_layout.addWidget(self.disk_usage)
        
        layout.addLayout(health_layout)

        # Data pipeline status
        self.pipeline_table = QTableWidget()
        self.pipeline_table.setColumnCount(4)
        self.pipeline_table.setHorizontalHeaderLabels(['Component', 'Status', 'Latency', 'Health'])
        layout.addWidget(self.pipeline_table)

        self.tab_widget.addTab(tab, "System Metrics")

    def create_alerts_tab(self):
        """Create alerts monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Alert controls
        control_layout = QHBoxLayout()
        
        self.alert_filter = QComboBox()
        self.alert_filter.addItems(['All', 'Critical', 'Warning', 'Info'])
        
        self.clear_alerts = QPushButton("Clear Alerts")
        
        control_layout.addWidget(QLabel("Filter:"))
        control_layout.addWidget(self.alert_filter)
        control_layout.addWidget(self.clear_alerts)
        
        layout.addLayout(control_layout)

        # Alerts table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(4)
        self.alerts_table.setHorizontalHeaderLabels(['Timestamp', 'Level', 'Component', 'Message'])
        layout.addWidget(self.alerts_table)

        self.tab_widget.addTab(tab, "Alerts")

    def create_performance_tab(self):
        """Create performance monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Performance charts
        self.performance_chart = PriceChart()
        layout.addWidget(self.performance_chart)

        # Performance metrics
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Current', 'Average'])
        layout.addWidget(self.metrics_table)

        self.tab_widget.addTab(tab, "Performance")

    def create_metric_widget(self, label):
        """Create metric monitoring widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label_widget = QLabel(label)
        progress = QProgressBar()
        progress.setRange(0, 100)
        
        layout.addWidget(label_widget)
        layout.addWidget(progress)
        
        return widget

    def setup_monitoring(self):
        """Setup monitoring connections"""
        self.event_system.register('system_metrics_update', self.update_system_metrics)
        self.event_system.register('alert_generated', self.update_alerts)
        self.event_system.register('performance_update', self.update_performance)

    def update_system_metrics(self, data):
        """Update system metrics display"""
        # Update CPU usage
        cpu_bar = self.cpu_usage.findChild(QProgressBar)
        cpu_bar.setValue(data['cpu_usage'])
        
        # Update memory usage
        memory_bar = self.memory_usage.findChild(QProgressBar)
        memory_bar.setValue(data['memory_usage'])
        
        # Update disk usage
        disk_bar = self.disk_usage.findChild(QProgressBar)
        disk_bar.setValue(data['disk_usage'])
        
        # Update pipeline status
        self.update_pipeline_table(data['pipeline_status'])

    def update_alerts(self, alert):
        """Update alerts display"""
        row = self.alerts_table.rowCount()
        self.alerts_table.insertRow(row)
        
        self.alerts_table.setItem(row, 0, QTableWidgetItem(alert['timestamp']))
        self.alerts_table.setItem(row, 1, QTableWidgetItem(alert['level']))
        self.alerts_table.setItem(row, 2, QTableWidgetItem(alert['component']))
        self.alerts_table.setItem(row, 3, QTableWidgetItem(alert['message']))

    def update_performance(self, data):
        """Update performance display"""
        self.performance_chart.update_data(data['performance_data'])
        self.update_metrics_table(data['metrics'])

    def update_pipeline_table(self, pipeline_status):
        """Update pipeline status table"""
        self.pipeline_table.setRowCount(len(pipeline_status))
        for i, status in enumerate(pipeline_status):
            self.pipeline_table.setItem(i, 0, QTableWidgetItem(status['component']))
            self.pipeline_table.setItem(i, 1, QTableWidgetItem(status['status']))
            self.pipeline_table.setItem(i, 2, QTableWidgetItem(str(status['latency'])))
            self.pipeline_table.setItem(i, 3, QTableWidgetItem(status['health']))

    def update_metrics_table(self, metrics):
        """Update metrics table"""
        self.metrics_table.setRowCount(len(metrics))
        for i, metric in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric['name']))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(metric['current'])))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(str(metric['average'])))
