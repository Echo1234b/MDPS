from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QTabWidget, QLabel, QTableWidget, QPushButton,
                           QComboBox, QCheckBox, QSpinBox)
from ..widgets.charts.price_chart import PriceChart

class ModelComparisonView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_comparison_tools()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Model Performance Tab
        self.create_performance_tab()

        # Prediction Analysis Tab
        self.create_prediction_tab()

        # Feature Importance Tab
        self.create_feature_tab()

    def create_performance_tab(self):
        """Create model performance comparison tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Model selection controls
        control_layout = QHBoxLayout()
        
        self.model_selector = QComboBox()
        self.model_selector.addItems(['All Models', 'LSTM', 'GRU', 'Transformer', 'Ensemble'])
        
        self.timeframe = QComboBox()
        self.timeframe.addItems(['1h', '4h', '1d', '1w'])
        
        self.compare_button = QPushButton('Compare')
        
        control_layout.addWidget(QLabel('Models:'))
        control_layout.addWidget(self.model_selector)
        control_layout.addWidget(QLabel('Timeframe:'))
        control_layout.addWidget(self.timeframe)
        control_layout.addWidget(self.compare_button)
        
        layout.addLayout(control_layout)

        # Performance metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])
        layout.addWidget(self.metrics_table)

        # Performance chart
        self.performance_chart = PriceChart()
        layout.addWidget(self.performance_chart)

        self.tab_widget.addTab(tab, "Performance")

    def create_prediction_tab(self):
        """Create prediction analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Prediction controls
        control_layout = QHBoxLayout()
        
        self.prediction_type = QComboBox()
        self.prediction_type.addItems(['Direction', 'Magnitude', 'Volatility'])
        
        self.confidence_threshold = QSpinBox()
        self.confidence_threshold.setRange(0, 100)
        self.confidence_threshold.setValue(70)
        
        self.analyze_button = QPushButton('Analyze')
        
        control_layout.addWidget(QLabel('Type:'))
        control_layout.addWidget(self.prediction_type)
        control_layout.addWidget(QLabel('Confidence %:'))
        control_layout.addWidget(self.confidence_threshold)
        control_layout.addWidget(self.analyze_button)
        
        layout.addLayout(control_layout)

        # Prediction comparison chart
        self.prediction_chart = PriceChart()
        layout.addWidget(self.prediction_chart)

        # Prediction statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(['Model', 'Correct', 'Incorrect', 'Accuracy'])
        layout.addWidget(self.stats_table)

        self.tab_widget.addTab(tab, "Predictions")

    def create_feature_tab(self):
        """Create feature importance comparison tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Feature selection controls
        control_layout = QHBoxLayout()
        
        self.feature_count = QSpinBox()
        self.feature_count.setRange(5, 50)
        self.feature_count.setValue(20)
        
        self.importance_method = QComboBox()
        self.importance_method.addItems(['SHAP', 'Permutation', 'Gain'])
        
        self.update_button = QPushButton('Update')
        
        control_layout.addWidget(QLabel('Top Features:'))
        control_layout.addWidget(self.feature_count)
        control_layout.addWidget(QLabel('Method:'))
        control_layout.addWidget(self.importance_method)
        control_layout.addWidget(self.update_button)
        
        layout.addLayout(control_layout)

        # Feature importance heatmap
        self.heatmap_view = PriceChart()
        layout.addWidget(self.heatmap_view)

        # Feature details table
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(4)
        self.feature_table.setHorizontalHeaderLabels(['Feature', 'Model 1', 'Model 2', 'Model 3'])
        layout.addWidget(self.feature_table)

        self.tab_widget.addTab(tab, "Features")

    def setup_comparison_tools(self):
        """Setup model comparison connections"""
        self.event_system.register('model_performance_update', self.update_performance)
        self.event_system.register('prediction_update', self.update_predictions)
        self.event_system.register('feature_importance_update', self.update_features)

        # Connect signals
        self.compare_button.clicked.connect(self.compare_models)
        self.analyze_button.clicked.connect(self.analyze_predictions)
        self.update_button.clicked.connect(self.update_features)

    def compare_models(self):
        """Compare selected models"""
        params = {
            'models': self.model_selector.currentText(),
            'timeframe': self.timeframe.currentText()
        }
        self.event_system.emit('compare_models', params)

    def analyze_predictions(self):
        """Analyze model predictions"""
        params = {
            'type': self.prediction_type.currentText(),
            'confidence': self.confidence_threshold.value()
        }
        self.event_system.emit('analyze_predictions', params)

    def update_features(self):
        """Update feature importance display"""
        params = {
            'feature_count': self.feature_count.value(),
            'method': self.importance_method.currentText()
        }
        self.event_system.emit('update_features', params)

    def update_performance(self, data):
        """Update performance comparison display"""
        self.metrics_table.setRowCount(len(data['metrics']))
        for i, metric in enumerate(data['metrics']):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric['model']))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(str(metric['accuracy'])))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(str(metric['precision'])))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(str(metric['recall'])))
            self.metrics_table.setItem(i, 4, QTableWidgetItem(str(metric['f1'])))
            self.metrics_table.setItem(i, 5, QTableWidgetItem(str(metric['auc'])))
        
        self.performance_chart.update_data(data['chart_data'])

    def update_predictions(self, data):
        """Update prediction analysis display"""
        self.prediction_chart.update_data(data['chart_data'])
        
        self.stats_table.setRowCount(len(data['stats']))
        for i, stat in enumerate(data['stats']):
            self.stats_table.setItem(i, 0, QTableWidgetItem(stat['model']))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(stat['correct'])))
            self.stats_table.setItem(i, 2, QTableWidgetItem(str(stat['incorrect'])))
            self.stats_table.setItem(i, 3, QTableWidgetItem(str(stat['accuracy'])))

    def update_features(self, data):
        """Update feature importance display"""
        self.heatmap_view.update_data(data['heatmap_data'])
        
        self.feature_table.setRowCount(len(data['features']))
        for i, feature in enumerate(data['features']):
            self.feature_table.setItem(i, 0, QTableWidgetItem(feature['name']))
            self.feature_table.setItem(i, 1, QTableWidgetItem(str(feature['model1'])))
            self.feature_table.setItem(i, 2, QTableWidgetItem(str(feature['model2'])))
            self.feature_table.setItem(i, 3, QTableWidgetItem(str(feature['model3'])))
