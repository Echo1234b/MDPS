from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QProgressBar, QTabWidget, QTableWidget)
from ..widgets.charts.price_chart import PriceChart

class AnalyticsView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_analytics_monitors()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Model Performance Tab
        self.create_model_performance_tab()

        # Prediction Analysis Tab
        self.create_prediction_analysis_tab()

        # Evaluation Metrics Tab
        self.create_evaluation_metrics_tab()

    def create_model_performance_tab(self):
        """Create model performance monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Model metrics section
        metrics_layout = QHBoxLayout()
        
        self.accuracy_label = QLabel("Model Accuracy:")
        self.accuracy_bar = QProgressBar()
        
        self.confidence_label = QLabel("Signal Confidence:")
        self.confidence_bar = QProgressBar()
        
        metrics_layout.addWidget(self.accuracy_label)
        metrics_layout.addWidget(self.accuracy_bar)
        metrics_layout.addWidget(self.confidence_label)
        metrics_layout.addWidget(self.confidence_bar)
        
        layout.addLayout(metrics_layout)

        # Performance chart
        self.performance_chart = PriceChart()
        layout.addWidget(self.performance_chart)

        self.tab_widget.addTab(tab, "Model Performance")

    def create_prediction_analysis_tab(self):
        """Create prediction analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Prediction confidence metrics
        confidence_layout = QHBoxLayout()
        
        self.class_prob = QLabel("Class Probability:")
        self.confidence_interval = QLabel("Confidence Interval:")
        self.certainty_score = QLabel("Certainty Score:")
        
        confidence_layout.addWidget(self.class_prob)
        confidence_layout.addWidget(self.confidence_interval)
        confidence_layout.addWidget(self.certainty_score)
        
        layout.addLayout(confidence_layout)

        # Feature importance table
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(3)
        self.feature_table.setHorizontalHeaderLabels(['Feature', 'Importance', 'Impact'])
        layout.addWidget(self.feature_table)

        self.tab_widget.addTab(tab, "Prediction Analysis")

    def create_evaluation_metrics_tab(self):
        """Create evaluation metrics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Performance metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Value', 'Target', 'Status'])
        layout.addWidget(self.metrics_table)

        self.tab_widget.addTab(tab, "Evaluation Metrics")

    def setup_analytics_monitors(self):
        """Setup analytics monitoring connections"""
        self.event_system.register('model_performance_update', self.update_model_performance)
        self.event_system.register('prediction_analysis_update', self.update_prediction_analysis)
        self.event_system.register('evaluation_metrics_update', self.update_evaluation_metrics)

    def update_model_performance(self, data):
        """Update model performance display"""
        self.accuracy_bar.setValue(int(data['accuracy'] * 100))
        self.confidence_bar.setValue(int(data['confidence'] * 100))
        self.performance_chart.update_data(data['performance_data'])

    def update_prediction_analysis(self, data):
        """Update prediction analysis display"""
        self.class_prob.setText(f"Class Probability: {data['class_prob']:.2f}")
        self.confidence_interval.setText(f"Confidence Interval: {data['conf_interval']}")
        self.certainty_score.setText(f"Certainty Score: {data['certainty_score']:.2f}")
        
        # Update feature importance table
        self.update_feature_table(data['feature_importance'])

    def update_evaluation_metrics(self, data):
        """Update evaluation metrics display"""
        self.metrics_table.setRowCount(len(data['metrics']))
        for i, metric in enumerate(data['metrics']):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric['name']))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(metric['value'])))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(str(metric['target'])))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(metric['status']))

    def update_feature_table(self, features):
        """Update feature importance table"""
        self.feature_table.setRowCount(len(features))
        for i, feature in enumerate(features):
            self.feature_table.setItem(i, 0, QTableWidgetItem(feature['name']))
            self.feature_table.setItem(i, 1, QTableWidgetItem(str(feature['importance'])))
            self.feature_table.setItem(i, 2, QTableWidgetItem(str(feature['impact'])))
