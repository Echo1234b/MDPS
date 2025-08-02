from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, 
                           QVBoxLayout, QStatusBar, QDockWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from .widgets.market_connection import MarketConnectionWidget
from .views.market_view import MarketView
from .views.technical_view import TechnicalView
from .views.trading_view import TradingView
from .views.analytics_view import AnalyticsView
from .views.system_monitor_view import SystemMonitorView
from .views.knowledge_graph_view import KnowledgeGraphView
from .views.external_factors_view import ExternalFactorsView
from .views.market_structure_view import MarketStructureView
from .views.pattern_recognition_view import PatternRecognitionView
from .views.model_comparison_view import ModelComparisonView
from .views.strategy_simulator_view import StrategySimulatorView
from .views.risk_management_view import RiskManagementView
from .views.data_quality_view import DataQualityView
from ..core.event_system import EventSystem

class MainWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.event_system = EventSystem()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Trading System')
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Initialize views and add tabs
        self.initialize_views()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add dock widgets
        self.create_dock_widgets()

    def initialize_views(self):
        """Initialize all views and add them as tabs"""
        # Create MT5 connection widget as the first tab
        self.market_connection = MarketConnectionWidget(config=self.config)
        self.tab_widget.addTab(self.market_connection, "MT5 Connection")
        
        # Define views and their tab labels
        view_configs = [
            (MarketView, "Market"),
            (TechnicalView, "Technical"),
            (TradingView, "Trading"),
            (AnalyticsView, "Analytics"),
            (SystemMonitorView, "System Monitor"),
            (KnowledgeGraphView, "Knowledge Graph"),
            (ExternalFactorsView, "External Factors"),
            (MarketStructureView, "Market Structure"),
            (PatternRecognitionView, "Pattern Recognition"),
            (ModelComparisonView, "Model Comparison"),
            (StrategySimulatorView, "Strategy Simulator"),
            (RiskManagementView, "Risk Management"),
            (DataQualityView, "Data Quality")
        ]

        # Create views and add tabs
        for view_class, tab_name in view_configs:
            view = view_class(self.event_system)
            # Store view instance as class attribute
            setattr(self, f"{tab_name.lower().replace(' ', '_')}_view", view)
            # Add view as a tab
            self.tab_widget.addTab(view, tab_name)

    def create_dock_widgets(self):
        """Create dock widgets for additional panels"""
        # Define dock widget configurations
        from PyQt5.QtCore import Qt
        dock_configs = [
            {
                "title": "Positions",
                "areas": Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea,
                "default": Qt.RightDockWidgetArea
            },
            {
                "title": "Orders",
                "areas": Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea,
                "default": Qt.BottomDockWidgetArea
            },
            {
                "title": "Alerts",
                "areas": Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea,
                "default": Qt.RightDockWidgetArea
            }
        ]

        # Create dock widgets
        for config in dock_configs:
            dock = QDockWidget(config["title"], self)
            dock.setAllowedAreas(config["areas"])
            self.addDockWidget(config["default"], dock)
