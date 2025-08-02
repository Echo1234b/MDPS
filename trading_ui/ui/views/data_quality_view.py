from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QTabWidget, QLabel, QTableWidget, QPushButton,
                           QComboBox, QProgressBar)
from ..widgets.charts.price_chart import PriceChart

class DataQualityView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_quality_monitors()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Data Validation Tab
        self.create_validation_tab()

        # Data Cleaning Tab
        self.create_cleaning_tab()

        # Data Pipeline Tab
        self.create_pipeline_tab()

    def create_validation_tab(self):
        """Create data validation monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Validation controls
        control_layout = QHBoxLayout()
        
        self.data_source = QComboBox()
        self.data_source.addItems(['All Sources', 'MT5', 'Binance', 'Coinbase'])
        
        self.validation_type = QComboBox()
        self.validation_type.addItems(['All Checks', 'Missing Values', 'Outliers', 'Anomalies'])
        
        self.run_validation = QPushButton('Run Validation')
        
        control_layout.addWidget(QLabel('Source:'))
        control_layout.addWidget(self.data_source)
        control_layout.addWidget(QLabel('Type:'))
        control_layout.addWidget(self.validation_type)
        control_layout.addWidget(self.run_validation)
        
        layout.addLayout(control_layout)

        # Validation metrics
        metrics_layout = QHBoxLayout()
        
        self.completeness = self.create_quality_metric("Completeness")
        self.accuracy = self.create_quality_metric("Accuracy")
        self.consistency = self.create_quality_metric("Consistency")
        
        metrics_layout.addWidget(self.completeness)
        metrics_layout.addWidget(self.accuracy)
        metrics_layout.addWidget(self.consistency)
        
        layout.addLayout(metrics_layout)

        # Validation results table
        self.validation_table = QTableWidget()
        self.validation_table.setColumnCount(5)
        self.validation_table.setHorizontalHeaderLabels(['Time', 'Source', 'Type', 'Status', 'Details'])
        layout.addWidget(self.validation_table)

        # Quality trend chart
        self.quality_chart = PriceChart()
        layout.addWidget(self.quality_chart)

        self.tab_widget.addTab(tab, "Validation")

    def create_cleaning_tab(self):
        """Create data cleaning monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Cleaning controls
        control_layout = QHBoxLayout()
        
        self.cleaning_method = QComboBox()
        self.cleaning_method.addItems(['All Methods', 'Interpolation', 'Smoothing', 'Outlier Removal'])
        
        self.apply_cleaning = QPushButton('Apply Cleaning')
        
        control_layout.addWidget(QLabel('Method:'))
        control_layout.addWidget(self.cleaning_method)
        control_layout.addWidget(self.apply_cleaning)
        
        layout.addLayout(control_layout)

        # Cleaning statistics
        stats_layout = QHBoxLayout()
        
        self.removed_outliers = self.create_quality_metric("Outliers Removed")
        self.imputed_values = self.create_quality_metric("Values Imputed")
        self.smoothed_points = self.create_quality_metric("Points Smoothed")
        
        stats_layout.addWidget(self.removed_outliers)
        stats_layout.addWidget(self.imputed_values)
        stats_layout.addWidget(self.smoothed_points)
        
        layout.addLayout(stats_layout)

        # Cleaning history table
        self.cleaning_table = QTableWidget()
        self.cleaning_table.setColumnCount(4)
        self.cleaning_table.setHorizontalHeaderLabels(['Time', 'Method', 'Records', 'Impact'])
        layout.addWidget(self.cleaning_table)

        self.tab_widget.addTab(tab, "Cleaning")

    def create_pipeline_tab(self):
        """Create data pipeline monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Pipeline controls
        control_layout = QHBoxLayout()
        
        self.pipeline_stage = QComboBox()
        self.pipeline_stage.addItems(['All Stages', 'Collection', 'Processing', 'Storage'])
        
        self.monitor_pipeline = QPushButton('Monitor')
        
        control_layout.addWidget(QLabel('Stage:'))
        control_layout.addWidget(self.pipeline_stage)
        control_layout.addWidget(self.monitor_pipeline)
        
        layout.addLayout(control_layout)

        # Pipeline status
        status_layout = QHBoxLayout()
        
        self.collection_status = self.create_pipeline_status("Collection")
        self.processing_status = self.create_pipeline_status("Processing")
        self.storage_status = self.create_pipeline_status("Storage")
        
        status_layout.addWidget(self.collection_status)
        status_layout.addWidget(self.processing_status)
        status_layout.addWidget(self.storage_status)
        
        layout.addLayout(status_layout)

        # Pipeline metrics table
        self.pipeline_table = QTableWidget()
        self.pipeline_table.setColumnCount(4)
        self.pipeline_table.setHorizontalHeaderLabels(['Stage', 'Status', 'Throughput', 'Latency'])
        layout.addWidget(self.pipeline_table)

        # Pipeline health chart
        self.pipeline_chart = PriceChart()
        layout.addWidget(self.pipeline_chart)

        self.tab_widget.addTab(tab, "Pipeline")

    def create_quality_metric(self, label):
        """Create quality metric widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label_widget = QLabel(label)
        value_label = QLabel("0%")
        progress = QProgressBar()
        progress.setRange(0, 100)
        
        layout.addWidget(label_widget)
        layout.addWidget(value_label)
        layout.addWidget(progress)
        
        return widget

    def create_pipeline_status(self, label):
        """Create pipeline status widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label_widget = QLabel(label)
        status_label = QLabel("Status: Unknown")
        health_bar = QProgressBar()
        health_bar.setRange(0, 100)
        
        layout.addWidget(label_widget)
        layout.addWidget(status_label)
        layout.addWidget(health_bar)
        
        return widget

    def setup_quality_monitors(self):
        """Setup quality monitoring connections"""
        self.event_system.register('validation_update', self.update_validation)
        self.event_system.register('cleaning_update', self.update_cleaning)
        self.event_system.register('pipeline_update', self.update_pipeline)

        # Connect signals
        self.run_validation.clicked.connect(self.run_data_validation)
        self.apply_cleaning.clicked.connect(self.apply_data_cleaning)
        self.monitor_pipeline.clicked.connect(self.monitor_data_pipeline)

    def run_data_validation(self):
        """Run data validation checks"""
        params = {
            'source': self.data_source.currentText(),
            'type': self.validation_type.currentText()
        }
        self.event_system.emit('run_validation', params)

    def apply_data_cleaning(self):
        """Apply data cleaning methods"""
        params = {
            'method': self.cleaning_method.currentText()
        }
        self.event_system.emit('apply_cleaning', params)

    def monitor_data_pipeline(self):
        """Monitor data pipeline health"""
        params = {
            'stage': self.pipeline_stage.currentText()
        }
        self.event_system.emit('monitor_pipeline', params)

    def update_validation(self, data):
        """Update validation display"""
        # Update quality metrics
        completeness_value = self.completeness.findChildren(QLabel)[1]
        completeness_progress = self.completeness.findChildren(QProgressBar)[0]
        completeness_value.setText(f"{data['completeness']:.1f}%")
        completeness_progress.setValue(int(data['completeness']))
        
        accuracy_value = self.accuracy.findChildren(QLabel)[1]
        accuracy_progress = self.accuracy.findChildren(QProgressBar)[0]
        accuracy_value.setText(f"{data['accuracy']:.1f}%")
        accuracy_progress.setValue(int(data['accuracy']))
        
        consistency_value = self.consistency.findChildren(QLabel)[1]
        consistency_progress = self.consistency.findChildren(QProgressBar)[0]
        consistency_value.setText(f"{data['consistency']:.1f}%")
        consistency_progress.setValue(int(data['consistency']))
        
        # Update validation table
        self.validation_table.setRowCount(len(data['validations']))
        for i, validation in enumerate(data['validations']):
            self.validation_table.setItem(i, 0, QTableWidgetItem(validation['time']))
            self.validation_table.setItem(i, 1, QTableWidgetItem(validation['source']))
            self.validation_table.setItem(i, 2, QTableWidgetItem(validation['type']))
            self.validation_table.setItem(i, 3, QTableWidgetItem(validation['status']))
            self.validation_table.setItem(i, 4, QTableWidgetItem(validation['details']))
        
        # Update quality chart
        self.quality_chart.update_data(data['quality_trend'])

    def update_cleaning(self, data):
        """Update cleaning display"""
        # Update cleaning statistics
        outliers_value = self.removed_outliers.findChildren(QLabel)[1]
        outliers_value.setText(str(data['outliers_removed']))
        
        imputed_value = self.imputed_values.findChildren(QLabel)[1]
        imputed_value.setText(str(data['values_imputed']))
        
        smoothed_value = self.smoothed_points.findChildren(QLabel)[1]
        smoothed_value.setText(str(data['points_smoothed']))
        
        # Update cleaning table
        self.cleaning_table.setRowCount(len(data['cleaning_history']))
        for i, cleaning in enumerate(data['cleaning_history']):
            self.cleaning_table.setItem(i, 0, QTableWidgetItem(cleaning['time']))
            self.cleaning_table.setItem(i, 1, QTableWidgetItem(cleaning['method']))
            self.cleaning_table.setItem(i, 2, QTableWidgetItem(str(cleaning['records'])))
            self.cleaning_table.setItem(i, 3, QTableWidgetItem(cleaning['impact']))

    def update_pipeline(self, data):
        """Update pipeline display"""
        # Update pipeline status
        for stage in ['collection', 'processing', 'storage']:
            status_widget = getattr(self, f"{stage}_status")
            status_label = status_widget.findChildren(QLabel)[1]
            health_bar = status_widget.findChildren(QProgressBar)[0]
            
            status_label.setText(f"Status: {data[stage]['status']}")
            health_bar.setValue(int(data[stage]['health']))
        
        # Update pipeline table
        self.pipeline_table.setRowCount(len(data['stages']))
        for i, stage in enumerate(data['stages']):
            self.pipeline_table.setItem(i, 0, QTableWidgetItem(stage['name']))
            self.pipeline_table.setItem(i, 1, QTableWidgetItem(stage['status']))
            self.pipeline_table.setItem(i, 2, QTableWidgetItem(str(stage['throughput'])))
            self.pipeline_table.setItem(i, 3, QTableWidgetItem(str(stage['latency'])))
        
        # Update pipeline chart
        self.pipeline_chart.update_data(data['health_trend'])
