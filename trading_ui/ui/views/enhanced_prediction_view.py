"""Enhanced Prediction Engine View with Model Comparison and Performance Metrics"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QGroupBox, QLabel, QProgressBar, QTextEdit, QTabWidget,
                           QTableWidget, QTableWidgetItem, QPushButton, QSplitter,
                           QListWidget, QListWidgetItem, QFrame, QScrollArea,
                           QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg
from .base_view import BaseView

class PredictionWorkerThread(QThread):
    """Thread for running prediction models"""
    
    prediction_ready = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.models = {}
        self.data = None
        
    def run(self):
        self.running = True
        while self.running:
            if self.data is not None:
                try:
                    # Simulate model predictions
                    predictions = self.generate_mock_predictions()
                    self.prediction_ready.emit(predictions)
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            self.msleep(5000)  # Run predictions every 5 seconds
    
    def generate_mock_predictions(self):
        """Generate mock predictions for demonstration"""
        models = ['LSTM', 'XGBoost', 'Random Forest', 'SVM', 'Neural Network']
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        predictions = {}
        for symbol in symbols:
            predictions[symbol] = {}
            for model in models:
                # Generate random predictions with some logic
                confidence = np.random.uniform(0.6, 0.95)
                direction = np.random.choice(['BUY', 'SELL', 'HOLD'])
                price_change = np.random.uniform(-0.002, 0.002)
                
                predictions[symbol][model] = {
                    'direction': direction,
                    'confidence': confidence,
                    'price_change': price_change,
                    'timestamp': datetime.now(),
                    'accuracy': np.random.uniform(0.65, 0.85),
                    'precision': np.random.uniform(0.70, 0.90),
                    'recall': np.random.uniform(0.60, 0.80)
                }
        
        return predictions
    
    def update_data(self, data):
        """Update data for predictions"""
        self.data = data
    
    def stop(self):
        self.running = False

class ModelPerformanceWidget(QWidget):
    """Widget for displaying model performance metrics"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Performance metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels([
            "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Last Update"
        ])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.metrics_table)
        
        # Performance chart
        self.performance_chart = pg.PlotWidget()
        self.performance_chart.setLabel('left', 'Accuracy (%)')
        self.performance_chart.setLabel('bottom', 'Time')
        self.performance_chart.addLegend()
        self.performance_chart.setMaximumHeight(200)
        
        layout.addWidget(self.performance_chart)
        
        # Initialize performance plots
        colors = ['r', 'g', 'b', 'c', 'm']
        self.performance_plots = {}
        models = ['LSTM', 'XGBoost', 'Random Forest', 'SVM', 'Neural Network']
        
        for i, model in enumerate(models):
            self.performance_plots[model] = self.performance_chart.plot(
                pen=colors[i % len(colors)], name=model
            )
    
    def update_performance(self, predictions):
        """Update performance display"""
        if not predictions:
            return
            
        # Update metrics table
        models = set()
        for symbol_predictions in predictions.values():
            models.update(symbol_predictions.keys())
        
        models = list(models)
        self.metrics_table.setRowCount(len(models))
        
        for i, model in enumerate(models):
            # Calculate average metrics across symbols
            accuracies = []
            precisions = []
            recalls = []
            
            for symbol_predictions in predictions.values():
                if model in symbol_predictions:
                    pred = symbol_predictions[model]
                    accuracies.append(pred.get('accuracy', 0))
                    precisions.append(pred.get('precision', 0))
                    recalls.append(pred.get('recall', 0))
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            avg_precision = np.mean(precisions) if precisions else 0
            avg_recall = np.mean(recalls) if recalls else 0
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            self.metrics_table.setItem(i, 0, QTableWidgetItem(model))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{avg_accuracy:.3f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{avg_precision:.3f}"))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{avg_recall:.3f}"))
            self.metrics_table.setItem(i, 4, QTableWidgetItem(f"{f1_score:.3f}"))
            self.metrics_table.setItem(i, 5, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))

class PredictionDisplayWidget(QWidget):
    """Widget for displaying current predictions"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Symbol tabs
        self.symbol_tabs = QTabWidget()
        layout.addWidget(self.symbol_tabs)
        
        # Initialize tabs for common symbols
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        self.symbol_widgets = {}
        
        for symbol in symbols:
            widget = self.create_symbol_widget(symbol)
            self.symbol_widgets[symbol] = widget
            self.symbol_tabs.addTab(widget, symbol)
    
    def create_symbol_widget(self, symbol):
        """Create widget for a specific symbol"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Current price and trend
        price_group = QGroupBox(f"{symbol} Current Status")
        price_layout = QGridLayout()
        
        price_label = QLabel("Price: 1.0000")
        price_label.setFont(QFont("Arial", 14, QFont.Bold))
        
        trend_label = QLabel("Trend: ↑")
        trend_label.setFont(QFont("Arial", 12))
        trend_label.setStyleSheet("color: green;")
        
        price_layout.addWidget(price_label, 0, 0)
        price_layout.addWidget(trend_label, 0, 1)
        
        price_group.setLayout(price_layout)
        layout.addWidget(price_group)
        
        # Model predictions table
        predictions_group = QGroupBox("Model Predictions")
        predictions_layout = QVBoxLayout()
        
        predictions_table = QTableWidget()
        predictions_table.setColumnCount(4)
        predictions_table.setHorizontalHeaderLabels([
            "Model", "Prediction", "Confidence", "Price Target"
        ])
        predictions_table.horizontalHeader().setStretchLastSection(True)
        predictions_table.setMaximumHeight(200)
        
        predictions_layout.addWidget(predictions_table)
        predictions_group.setLayout(predictions_layout)
        layout.addWidget(predictions_group)
        
        # Consensus prediction
        consensus_group = QGroupBox("Consensus Prediction")
        consensus_layout = QGridLayout()
        
        consensus_direction = QLabel("Direction: HOLD")
        consensus_direction.setFont(QFont("Arial", 12, QFont.Bold))
        
        consensus_confidence = QLabel("Confidence: 0%")
        consensus_strength = QProgressBar()
        
        consensus_layout.addWidget(QLabel("Consensus:"), 0, 0)
        consensus_layout.addWidget(consensus_direction, 0, 1)
        consensus_layout.addWidget(QLabel("Confidence:"), 1, 0)
        consensus_layout.addWidget(consensus_confidence, 1, 1)
        consensus_layout.addWidget(QLabel("Strength:"), 2, 0)
        consensus_layout.addWidget(consensus_strength, 2, 1)
        
        consensus_group.setLayout(consensus_layout)
        layout.addWidget(consensus_group)
        
        # Store references for updates
        setattr(widget, 'price_label', price_label)
        setattr(widget, 'trend_label', trend_label)
        setattr(widget, 'predictions_table', predictions_table)
        setattr(widget, 'consensus_direction', consensus_direction)
        setattr(widget, 'consensus_confidence', consensus_confidence)
        setattr(widget, 'consensus_strength', consensus_strength)
        
        return widget
    
    def update_predictions(self, predictions):
        """Update prediction displays"""
        for symbol, symbol_predictions in predictions.items():
            if symbol in self.symbol_widgets:
                widget = self.symbol_widgets[symbol]
                self.update_symbol_widget(widget, symbol, symbol_predictions)
    
    def update_symbol_widget(self, widget, symbol, predictions):
        """Update a specific symbol widget"""
        # Update predictions table
        table = widget.predictions_table
        table.setRowCount(len(predictions))
        
        directions = []
        confidences = []
        
        for i, (model, pred) in enumerate(predictions.items()):
            table.setItem(i, 0, QTableWidgetItem(model))
            
            direction = pred.get('direction', 'HOLD')
            table.setItem(i, 1, QTableWidgetItem(direction))
            directions.append(direction)
            
            confidence = pred.get('confidence', 0)
            table.setItem(i, 2, QTableWidgetItem(f"{confidence:.2%}"))
            confidences.append(confidence)
            
            price_change = pred.get('price_change', 0)
            table.setItem(i, 3, QTableWidgetItem(f"{price_change:+.5f}"))
            
            # Color code by direction
            direction_color = QColor("green") if direction == "BUY" else QColor("red") if direction == "SELL" else QColor("gray")
            table.item(i, 1).setForeground(direction_color)
        
        # Calculate consensus
        if directions:
            # Most common direction
            consensus_dir = max(set(directions), key=directions.count)
            avg_confidence = np.mean(confidences)
            
            widget.consensus_direction.setText(f"Direction: {consensus_dir}")
            widget.consensus_confidence.setText(f"Confidence: {avg_confidence:.1%}")
            widget.consensus_strength.setValue(int(avg_confidence * 100))
            
            # Color code consensus
            consensus_color = "green" if consensus_dir == "BUY" else "red" if consensus_dir == "SELL" else "gray"
            widget.consensus_direction.setStyleSheet(f"color: {consensus_color}; font-weight: bold;")

class EnhancedPredictionView(BaseView):
    """Enhanced prediction engine view with comprehensive model management"""
    
    def __init__(self, event_system):
        super().__init__(event_system)
        self.prediction_thread = None
        self.prediction_history = {}
        self.init_ui()
        self.setup_prediction_engine()
        
    def init_ui(self):
        """Initialize the enhanced UI"""
        layout = QVBoxLayout(self)
        
        # Main tab widget
        main_tabs = QTabWidget()
        layout.addWidget(main_tabs)
        
        # Predictions tab
        predictions_tab = self.create_predictions_tab()
        main_tabs.addTab(predictions_tab, "Live Predictions")
        
        # Model Management tab
        models_tab = self.create_models_tab()
        main_tabs.addTab(models_tab, "Model Management")
        
        # Performance Analysis tab
        performance_tab = self.create_performance_tab()
        main_tabs.addTab(performance_tab, "Performance Analysis")
        
        # Settings tab
        settings_tab = self.create_settings_tab()
        main_tabs.addTab(settings_tab, "Settings")
    
    def create_predictions_tab(self):
        """Create the live predictions tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Main prediction display
        self.prediction_display = PredictionDisplayWidget()
        layout.addWidget(self.prediction_display, 3)
        
        # Side panel with controls and summary
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        
        # Control buttons
        controls_group = QGroupBox("Prediction Controls")
        controls_layout = QVBoxLayout()
        
        self.start_predictions_btn = QPushButton("Start Predictions")
        self.start_predictions_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.stop_predictions_btn = QPushButton("Stop Predictions")
        self.stop_predictions_btn.setEnabled(False)
        
        self.refresh_btn = QPushButton("Refresh Now")
        
        controls_layout.addWidget(self.start_predictions_btn)
        controls_layout.addWidget(self.stop_predictions_btn)
        controls_layout.addWidget(self.refresh_btn)
        
        controls_group.setLayout(controls_layout)
        side_layout.addWidget(controls_group)
        
        # Prediction summary
        summary_group = QGroupBox("Prediction Summary")
        summary_layout = QGridLayout()
        
        self.total_predictions_label = QLabel("Total: 0")
        self.bullish_predictions_label = QLabel("Bullish: 0")
        self.bearish_predictions_label = QLabel("Bearish: 0")
        self.neutral_predictions_label = QLabel("Neutral: 0")
        
        summary_layout.addWidget(self.total_predictions_label, 0, 0)
        summary_layout.addWidget(self.bullish_predictions_label, 1, 0)
        summary_layout.addWidget(self.bearish_predictions_label, 2, 0)
        summary_layout.addWidget(self.neutral_predictions_label, 3, 0)
        
        summary_group.setLayout(summary_layout)
        side_layout.addWidget(summary_group)
        
        # Market sentiment gauge
        sentiment_group = QGroupBox("Market Sentiment")
        sentiment_layout = QVBoxLayout()
        
        self.sentiment_gauge = QProgressBar()
        self.sentiment_gauge.setRange(-100, 100)
        self.sentiment_gauge.setValue(0)
        self.sentiment_gauge.setFormat("Neutral")
        
        sentiment_layout.addWidget(self.sentiment_gauge)
        sentiment_group.setLayout(sentiment_layout)
        side_layout.addWidget(sentiment_group)
        
        side_layout.addStretch()
        layout.addWidget(side_panel, 1)
        
        return widget
    
    def create_models_tab(self):
        """Create the model management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model list and controls
        model_controls_layout = QHBoxLayout()
        
        # Model list
        models_group = QGroupBox("Available Models")
        models_layout = QVBoxLayout()
        
        self.models_list = QListWidget()
        models = [
            "LSTM Neural Network",
            "XGBoost Classifier",
            "Random Forest",
            "Support Vector Machine",
            "Deep Neural Network",
            "Transformer Model"
        ]
        
        for model in models:
            item = QListWidgetItem(f"● {model}")
            item.setForeground(QColor("green"))
            self.models_list.addItem(item)
        
        models_layout.addWidget(self.models_list)
        models_group.setLayout(models_layout)
        model_controls_layout.addWidget(models_group, 2)
        
        # Model details and controls
        details_group = QGroupBox("Model Details")
        details_layout = QVBoxLayout()
        
        # Model info
        self.model_name_label = QLabel("Model: Not Selected")
        self.model_status_label = QLabel("Status: Inactive")
        self.model_accuracy_label = QLabel("Accuracy: N/A")
        self.model_last_trained_label = QLabel("Last Trained: N/A")
        
        details_layout.addWidget(self.model_name_label)
        details_layout.addWidget(self.model_status_label)
        details_layout.addWidget(self.model_accuracy_label)
        details_layout.addWidget(self.model_last_trained_label)
        
        # Model controls
        model_action_layout = QHBoxLayout()
        
        self.train_model_btn = QPushButton("Train Model")
        self.test_model_btn = QPushButton("Test Model")
        self.deploy_model_btn = QPushButton("Deploy Model")
        
        model_action_layout.addWidget(self.train_model_btn)
        model_action_layout.addWidget(self.test_model_btn)
        model_action_layout.addWidget(self.deploy_model_btn)
        
        details_layout.addLayout(model_action_layout)
        details_group.setLayout(details_layout)
        model_controls_layout.addWidget(details_group, 1)
        
        layout.addLayout(model_controls_layout)
        
        # Model comparison
        comparison_group = QGroupBox("Model Comparison")
        comparison_layout = QVBoxLayout()
        
        self.model_comparison_table = QTableWidget()
        self.model_comparison_table.setColumnCount(6)
        self.model_comparison_table.setHorizontalHeaderLabels([
            "Model", "Accuracy", "Training Time", "Prediction Speed", "Memory Usage", "Status"
        ])
        self.model_comparison_table.horizontalHeader().setStretchLastSection(True)
        
        comparison_layout.addWidget(self.model_comparison_table)
        comparison_group.setLayout(comparison_layout)
        layout.addWidget(comparison_group)
        
        return widget
    
    def create_performance_tab(self):
        """Create the performance analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance metrics widget
        self.performance_widget = ModelPerformanceWidget()
        layout.addWidget(self.performance_widget)
        
        return widget
    
    def create_settings_tab(self):
        """Create the settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Prediction settings
        pred_settings_group = QGroupBox("Prediction Settings")
        pred_settings_layout = QGridLayout()
        
        # Prediction interval
        pred_settings_layout.addWidget(QLabel("Prediction Interval:"), 0, 0)
        self.pred_interval_spin = QSpinBox()
        self.pred_interval_spin.setRange(1, 300)
        self.pred_interval_spin.setValue(30)
        self.pred_interval_spin.setSuffix(" seconds")
        pred_settings_layout.addWidget(self.pred_interval_spin, 0, 1)
        
        # Confidence threshold
        pred_settings_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.1, 1.0)
        self.confidence_threshold_spin.setValue(0.7)
        self.confidence_threshold_spin.setSingleStep(0.1)
        pred_settings_layout.addWidget(self.confidence_threshold_spin, 1, 1)
        
        # Enable/disable models
        pred_settings_layout.addWidget(QLabel("Model Selection:"), 2, 0)
        
        models_selection_layout = QVBoxLayout()
        self.model_checkboxes = {}
        models = ['LSTM', 'XGBoost', 'Random Forest', 'SVM', 'Neural Network']
        
        for model in models:
            checkbox = QCheckBox(model)
            checkbox.setChecked(True)
            self.model_checkboxes[model] = checkbox
            models_selection_layout.addWidget(checkbox)
        
        pred_settings_layout.addLayout(models_selection_layout, 2, 1)
        
        pred_settings_group.setLayout(pred_settings_layout)
        layout.addWidget(pred_settings_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QGridLayout()
        
        # Ensemble method
        advanced_layout.addWidget(QLabel("Ensemble Method:"), 0, 0)
        self.ensemble_combo = QComboBox()
        self.ensemble_combo.addItems(["Voting", "Weighted Average", "Stacking"])
        advanced_layout.addWidget(self.ensemble_combo, 0, 1)
        
        # Feature importance threshold
        advanced_layout.addWidget(QLabel("Feature Importance Threshold:"), 1, 0)
        self.feature_importance_slider = QSlider(Qt.Horizontal)
        self.feature_importance_slider.setRange(1, 100)
        self.feature_importance_slider.setValue(50)
        advanced_layout.addWidget(self.feature_importance_slider, 1, 1)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Save settings button
        save_btn = QPushButton("Save Settings")
        layout.addWidget(save_btn)
        
        layout.addStretch()
        
        return widget
    
    def setup_prediction_engine(self):
        """Setup the prediction engine"""
        # Create and start prediction thread
        self.prediction_thread = PredictionWorkerThread()
        self.prediction_thread.prediction_ready.connect(self.update_predictions)
        
        # Connect control buttons
        self.start_predictions_btn.clicked.connect(self.start_predictions)
        self.stop_predictions_btn.clicked.connect(self.stop_predictions)
        self.refresh_btn.clicked.connect(self.manual_refresh)
        
        # Connect model list selection
        self.models_list.itemClicked.connect(self.on_model_selected)
    
    def start_predictions(self):
        """Start the prediction engine"""
        if not self.prediction_thread.isRunning():
            self.prediction_thread.start()
        
        self.start_predictions_btn.setEnabled(False)
        self.stop_predictions_btn.setEnabled(True)
    
    def stop_predictions(self):
        """Stop the prediction engine"""
        if self.prediction_thread.isRunning():
            self.prediction_thread.stop()
            self.prediction_thread.wait()
        
        self.start_predictions_btn.setEnabled(True)
        self.stop_predictions_btn.setEnabled(False)
    
    def manual_refresh(self):
        """Manually refresh predictions"""
        if self.prediction_thread.isRunning():
            # Trigger immediate prediction
            pass
    
    @pyqtSlot(dict)
    def update_predictions(self, predictions):
        """Update prediction displays"""
        # Update main prediction display
        self.prediction_display.update_predictions(predictions)
        
        # Update performance widget
        self.performance_widget.update_performance(predictions)
        
        # Update summary
        self.update_prediction_summary(predictions)
    
    def update_prediction_summary(self, predictions):
        """Update the prediction summary panel"""
        total_predictions = 0
        bullish = 0
        bearish = 0
        neutral = 0
        
        for symbol_predictions in predictions.values():
            for pred in symbol_predictions.values():
                total_predictions += 1
                direction = pred.get('direction', 'HOLD')
                if direction == 'BUY':
                    bullish += 1
                elif direction == 'SELL':
                    bearish += 1
                else:
                    neutral += 1
        
        self.total_predictions_label.setText(f"Total: {total_predictions}")
        self.bullish_predictions_label.setText(f"Bullish: {bullish}")
        self.bearish_predictions_label.setText(f"Bearish: {bearish}")
        self.neutral_predictions_label.setText(f"Neutral: {neutral}")
        
        # Update sentiment gauge
        if total_predictions > 0:
            sentiment_score = ((bullish - bearish) / total_predictions) * 100
            self.sentiment_gauge.setValue(int(sentiment_score))
            
            if sentiment_score > 20:
                self.sentiment_gauge.setFormat("Bullish")
            elif sentiment_score < -20:
                self.sentiment_gauge.setFormat("Bearish")
            else:
                self.sentiment_gauge.setFormat("Neutral")
    
    def on_model_selected(self, item):
        """Handle model selection"""
        model_name = item.text().replace("● ", "")
        self.model_name_label.setText(f"Model: {model_name}")
        self.model_status_label.setText("Status: Active")
        self.model_accuracy_label.setText("Accuracy: 82.5%")
        self.model_last_trained_label.setText("Last Trained: 2024-01-15")
    
    def update_data(self, data):
        """Update view with new MDPS data"""
        if self.prediction_thread:
            self.prediction_thread.update_data(data)
    
    def closeEvent(self, event):
        """Clean up when closing"""
        if self.prediction_thread:
            self.prediction_thread.stop()
            self.prediction_thread.wait()
        event.accept()