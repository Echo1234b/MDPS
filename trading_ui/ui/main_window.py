from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, 
                           QVBoxLayout, QStatusBar, QDockWidget, QMenuBar, 
                           QMenu, QAction, QMessageBox, QSplitter, QToolBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from .widgets.enhanced_market_connection import EnhancedMarketConnectionWidget
from .views.market_view import MarketView
from .views.technical_view import TechnicalView
from .views.trading_view import TradingView
from .views.analytics_view import AnalyticsView
from .views.enhanced_system_monitor_view import EnhancedSystemMonitorView
from .views.knowledge_graph_view import KnowledgeGraphView
from .views.external_factors_view import ExternalFactorsView
from .views.market_structure_view import MarketStructureView
from .views.pattern_recognition_view import PatternRecognitionView
from .views.model_comparison_view import ModelComparisonView
from .views.strategy_simulator_view import StrategySimulatorView
from .views.risk_management_view import RiskManagementView
from .views.data_quality_view import DataQualityView
from ..core.event_system import EventSystem
from ..core.mdps_controller import MDPSController

class MainWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.event_system = EventSystem()
        
        # Initialize MDPS controller
        self.mdps_controller = MDPSController(config, self.event_system)
        self.setup_mdps_connections()
        
        # Initialize UI
        self.init_ui()
        
        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_all_views)
        self.refresh_timer.start(1000)  # Refresh every second

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('MDPS - Multi-Dimensional Prediction System')
        self.setGeometry(50, 50, 1600, 1000)

        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()

        # Create main splitter for layout
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_splitter.addWidget(self.tab_widget)

        # Initialize views and add tabs
        self.initialize_views()

        # Create status bar with MDPS status
        self.create_enhanced_status_bar()

        # Add dock widgets
        self.create_dock_widgets()
        
        # Set splitter sizes (80% for main area, 20% for sidebar)
        main_splitter.setSizes([1280, 320])

    def initialize_views(self):
        """Initialize all views and add them as tabs"""
        # Create enhanced MT5 connection widget as the first tab
        self.market_connection = EnhancedMarketConnectionWidget(config=self.config)
        self.tab_widget.addTab(self.market_connection, "MT5 Connection & Control")
        
        # Define views and their tab labels
        view_configs = [
            (MarketView, "Market"),
            (TechnicalView, "Technical"),
            (TradingView, "Trading"),
            (AnalyticsView, "Analytics"),
            (EnhancedSystemMonitorView, "System Monitor"),
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
    
    def create_menu_bar(self):
        """Create enhanced menu bar with MDPS controls"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('&File')
        
        # MDPS Actions
        start_mdps_action = QAction('&Start MDPS', self)
        start_mdps_action.setShortcut(QKeySequence('Ctrl+S'))
        start_mdps_action.triggered.connect(self.start_mdps)
        file_menu.addAction(start_mdps_action)
        
        stop_mdps_action = QAction('S&top MDPS', self)
        stop_mdps_action.setShortcut(QKeySequence('Ctrl+T'))
        stop_mdps_action.triggered.connect(self.stop_mdps)
        file_menu.addAction(stop_mdps_action)
        
        file_menu.addSeparator()
        
        # Configuration
        config_action = QAction('&Configuration', self)
        config_action.setShortcut(QKeySequence('Ctrl+P'))
        config_action.triggered.connect(self.show_configuration)
        file_menu.addAction(config_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence('Ctrl+Q'))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View Menu
        view_menu = menubar.addMenu('&View')
        
        # Toggle full screen
        fullscreen_action = QAction('&Full Screen', self)
        fullscreen_action.setShortcut(QKeySequence('F11'))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Refresh
        refresh_action = QAction('&Refresh All', self)
        refresh_action.setShortcut(QKeySequence('F5'))
        refresh_action.triggered.connect(self.refresh_all_views)
        view_menu.addAction(refresh_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu('&Tools')
        
        # System Monitor
        monitor_action = QAction('&System Monitor', self)
        monitor_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(4))
        tools_menu.addAction(monitor_action)
        
        # Data Quality Check
        quality_action = QAction('&Data Quality Check', self)
        quality_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(12))
        tools_menu.addAction(quality_action)
        
        # Help Menu
        help_menu = menubar.addMenu('&Help')
        
        # About
        about_action = QAction('&About MDPS', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create toolbar with quick access buttons"""
        toolbar = QToolBar('Main Toolbar')
        self.addToolBar(toolbar)
        
        # MDPS Controls
        start_action = QAction('Start MDPS', self)
        start_action.triggered.connect(self.start_mdps)
        toolbar.addAction(start_action)
        
        stop_action = QAction('Stop MDPS', self)
        stop_action.triggered.connect(self.stop_mdps)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        # Quick navigation
        market_action = QAction('Market', self)
        market_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        toolbar.addAction(market_action)
        
        trading_action = QAction('Trading', self)
        trading_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(3))
        toolbar.addAction(trading_action)
        
        analytics_action = QAction('Analytics', self)
        analytics_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(4))
        toolbar.addAction(analytics_action)
    
    def create_enhanced_status_bar(self):
        """Create enhanced status bar with MDPS status indicators"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # MDPS status label
        self.mdps_status_label = QWidget()
        self.mdps_status_label.setFixedWidth(200)
        self.status_bar.addPermanentWidget(self.mdps_status_label)
        
        # Connection status
        self.connection_status_label = QWidget()
        self.connection_status_label.setFixedWidth(150)
        self.status_bar.addPermanentWidget(self.connection_status_label)
        
        # Default status
        self.status_bar.showMessage("MDPS Ready - Click Start to begin processing")
    
    def setup_mdps_connections(self):
        """Setup connections between MDPS controller and UI"""
        # Connect MDPS controller signals
        self.mdps_controller.data_updated.connect(self.on_mdps_data_updated)
        self.mdps_controller.status_changed.connect(self.on_mdps_status_changed)
        self.mdps_controller.error_occurred.connect(self.on_mdps_error)
        self.mdps_controller.connection_status_changed.connect(self.on_connection_status_changed)
    
    def start_mdps(self):
        """Start MDPS processing"""
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]  # Can be made configurable
        timeframe = "M5"
        update_interval = 300
        
        success = self.mdps_controller.start_processing(symbols, timeframe, update_interval)
        if success:
            self.status_bar.showMessage("Starting MDPS processing...")
        else:
            QMessageBox.warning(self, "MDPS Error", "Failed to start MDPS processing")
    
    def stop_mdps(self):
        """Stop MDPS processing"""
        self.mdps_controller.stop_processing()
        self.status_bar.showMessage("MDPS processing stopped")
    
    def on_mdps_data_updated(self, data):
        """Handle MDPS data updates"""
        # Update all views with new data
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if hasattr(widget, 'update_data'):
                widget.update_data(data)
    
    def on_mdps_status_changed(self, status):
        """Handle MDPS status changes"""
        self.status_bar.showMessage(f"MDPS: {status}")
    
    def on_mdps_error(self, error):
        """Handle MDPS errors"""
        QMessageBox.critical(self, "MDPS Error", f"Error: {error}")
        self.status_bar.showMessage(f"MDPS Error: {error}")
    
    def on_connection_status_changed(self, connected):
        """Handle connection status changes"""
        if connected:
            self.status_bar.showMessage("MDPS: Connected and Processing")
        else:
            self.status_bar.showMessage("MDPS: Disconnected")
    
    def refresh_all_views(self):
        """Refresh all views with latest data"""
        current_results = self.mdps_controller.get_current_results()
        if current_results:
            self.on_mdps_data_updated(current_results)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def show_configuration(self):
        """Show configuration dialog"""
        # TODO: Implement configuration dialog
        QMessageBox.information(self, "Configuration", "Configuration dialog will be implemented")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About MDPS", 
                         "Multi-Dimensional Prediction System\n\n"
                         "Advanced trading system with integrated ML models,\n"
                         "technical analysis, and real-time market processing.\n\n"
                         "Version 2.0\n"
                         "© 2024 MDPS Development Team")
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop MDPS processing before closing
        if self.mdps_controller.is_running:
            self.mdps_controller.stop_processing()
        
        # Stop refresh timer
        self.refresh_timer.stop()
        
        # Accept close event
        event.accept()
