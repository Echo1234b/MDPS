"""Enhanced Market Connection Widget with MDPS Integration"""
import sys
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QComboBox, QGridLayout, QGroupBox, QLineEdit,
                           QTextEdit, QProgressBar, QCheckBox, QSpinBox, QTabWidget,
                           QTableWidget, QTableWidgetItem, QHeaderView, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette

# Add project root to Python path
root_path = str(Path(__file__).parent.parent.parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from Data_Collection_and_Acquisition import MT5ConnectionManager
except ImportError:
    # Fallback for development
    class MT5ConnectionManager:
        def __init__(self, config=None):
            self.config = config
            self.connected = False
        
        def connect(self, login, password, server):
            return True
        
        def disconnect(self):
            return True
        
        def get_symbols(self):
            return ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "AUDUSD"]
        
        def get_account_info(self):
            return {"balance": 10000, "equity": 10000, "margin": 0}

class ConnectionMonitorThread(QThread):
    """Thread for monitoring MT5 connection status"""
    
    status_update = pyqtSignal(dict)
    
    def __init__(self, mt5_manager):
        super().__init__()
        self.mt5_manager = mt5_manager
        self.running = False
        
    def run(self):
        self.running = True
        while self.running:
            if self.mt5_manager:
                # Get connection status and account info
                status = {
                    'connected': getattr(self.mt5_manager, 'connected', False),
                    'account_info': self.mt5_manager.get_account_info() if hasattr(self.mt5_manager, 'get_account_info') else {},
                    'symbols_count': len(self.mt5_manager.get_symbols() if hasattr(self.mt5_manager, 'get_symbols') else [])
                }
                self.status_update.emit(status)
            
            self.msleep(1000)  # Update every second
    
    def stop(self):
        self.running = False

class EnhancedMarketConnectionWidget(QWidget):
    """Enhanced market connection widget with advanced features"""
    
    connection_status_changed = pyqtSignal(bool)
    data_received = pyqtSignal(dict)
    
    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.mt5_manager = MT5ConnectionManager(config=self.config)
        self.monitor_thread = None
        self.auto_reconnect = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        self.init_ui()
        self.setup_connections()
        self.setup_monitoring()
        
    def init_ui(self):
        """Initialize the enhanced UI components"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for organized layout
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Connection tab
        self.create_connection_tab()
        
        # Monitoring tab
        self.create_monitoring_tab()
        
        # Settings tab
        self.create_settings_tab()
        
    def create_connection_tab(self):
        """Create the main connection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Connection group
        connection_group = QGroupBox("MetaTrader 5 Connection")
        connection_layout = QGridLayout()
        
        # Connection controls
        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText("Login ID")
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        
        self.server_input = QLineEdit()
        self.server_input.setPlaceholderText("Server")
        
        if self.config and hasattr(self.config, 'mt5_settings'):
            self.server_input.setText(self.config.mt5_settings.get("server", ""))
            
        # Connection buttons
        self.connect_button = QPushButton("Connect to MT5")
        self.connect_button.setStyleSheet("""
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
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)
        self.disconnect_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c1170b;
            }
        """)
        
        # Status indicator
        self.status_label = QLabel("Not Connected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        # Progress bar for connection
        self.connection_progress = QProgressBar()
        self.connection_progress.setVisible(False)
        
        # Add widgets to layout
        connection_layout.addWidget(QLabel("Login:"), 0, 0)
        connection_layout.addWidget(self.login_input, 0, 1)
        connection_layout.addWidget(QLabel("Password:"), 1, 0)
        connection_layout.addWidget(self.password_input, 1, 1)
        connection_layout.addWidget(QLabel("Server:"), 2, 0)
        connection_layout.addWidget(self.server_input, 2, 1)
        connection_layout.addWidget(self.status_label, 3, 0, 1, 2)
        connection_layout.addWidget(self.connection_progress, 4, 0, 1, 2)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.connect_button)
        button_layout.addWidget(self.disconnect_button)
        connection_layout.addLayout(button_layout, 5, 0, 1, 2)
        
        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)
        
        # Symbols group
        symbols_group = QGroupBox("Available Symbols")
        symbols_layout = QVBoxLayout()
        
        self.symbols_table = QTableWidget()
        self.symbols_table.setColumnCount(4)
        self.symbols_table.setHorizontalHeaderLabels(["Symbol", "Bid", "Ask", "Spread"])
        self.symbols_table.horizontalHeader().setStretchLastSection(True)
        symbols_layout.addWidget(self.symbols_table)
        
        symbols_group.setLayout(symbols_layout)
        layout.addWidget(symbols_group)
        
        self.tab_widget.addTab(tab, "Connection")
    
    def create_monitoring_tab(self):
        """Create the monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Account info group
        account_group = QGroupBox("Account Information")
        account_layout = QGridLayout()
        
        self.balance_label = QLabel("Balance: $0.00")
        self.equity_label = QLabel("Equity: $0.00")
        self.margin_label = QLabel("Margin: $0.00")
        self.free_margin_label = QLabel("Free Margin: $0.00")
        
        account_layout.addWidget(QLabel("Account Status:"), 0, 0)
        account_layout.addWidget(self.balance_label, 1, 0)
        account_layout.addWidget(self.equity_label, 1, 1)
        account_layout.addWidget(self.margin_label, 2, 0)
        account_layout.addWidget(self.free_margin_label, 2, 1)
        
        account_group.setLayout(account_layout)
        layout.addWidget(account_group)
        
        # Connection stats group
        stats_group = QGroupBox("Connection Statistics")
        stats_layout = QGridLayout()
        
        self.uptime_label = QLabel("Uptime: 00:00:00")
        self.ping_label = QLabel("Ping: 0ms")
        self.reconnects_label = QLabel("Reconnects: 0")
        self.data_packets_label = QLabel("Data Packets: 0")
        
        stats_layout.addWidget(self.uptime_label, 0, 0)
        stats_layout.addWidget(self.ping_label, 0, 1)
        stats_layout.addWidget(self.reconnects_label, 1, 0)
        stats_layout.addWidget(self.data_packets_label, 1, 1)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Log area
        log_group = QGroupBox("Connection Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.tab_widget.addTab(tab, "Monitoring")
    
    def create_settings_tab(self):
        """Create the settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Connection settings
        connection_settings_group = QGroupBox("Connection Settings")
        settings_layout = QGridLayout()
        
        # Auto-reconnect
        self.auto_reconnect_checkbox = QCheckBox("Auto Reconnect")
        self.auto_reconnect_checkbox.setChecked(True)
        
        # Reconnect attempts
        self.reconnect_attempts_spin = QSpinBox()
        self.reconnect_attempts_spin.setRange(1, 10)
        self.reconnect_attempts_spin.setValue(5)
        
        # Connection timeout
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 120)
        self.timeout_spin.setValue(30)
        self.timeout_spin.setSuffix(" seconds")
        
        # Update frequency
        self.update_frequency_spin = QSpinBox()
        self.update_frequency_spin.setRange(1, 60)
        self.update_frequency_spin.setValue(1)
        self.update_frequency_spin.setSuffix(" seconds")
        
        settings_layout.addWidget(self.auto_reconnect_checkbox, 0, 0, 1, 2)
        settings_layout.addWidget(QLabel("Max Reconnect Attempts:"), 1, 0)
        settings_layout.addWidget(self.reconnect_attempts_spin, 1, 1)
        settings_layout.addWidget(QLabel("Connection Timeout:"), 2, 0)
        settings_layout.addWidget(self.timeout_spin, 2, 1)
        settings_layout.addWidget(QLabel("Update Frequency:"), 3, 0)
        settings_layout.addWidget(self.update_frequency_spin, 3, 1)
        
        connection_settings_group.setLayout(settings_layout)
        layout.addWidget(connection_settings_group)
        
        # Data settings
        data_settings_group = QGroupBox("Data Settings")
        data_layout = QGridLayout()
        
        # Symbol filter
        self.symbol_filter_input = QLineEdit()
        self.symbol_filter_input.setPlaceholderText("Filter symbols (e.g., EUR, USD)")
        
        # Enable/disable data streams
        self.enable_tick_data = QCheckBox("Enable Tick Data")
        self.enable_tick_data.setChecked(True)
        
        self.enable_market_depth = QCheckBox("Enable Market Depth")
        self.enable_market_depth.setChecked(False)
        
        data_layout.addWidget(QLabel("Symbol Filter:"), 0, 0)
        data_layout.addWidget(self.symbol_filter_input, 0, 1)
        data_layout.addWidget(self.enable_tick_data, 1, 0, 1, 2)
        data_layout.addWidget(self.enable_market_depth, 2, 0, 1, 2)
        
        data_settings_group.setLayout(data_layout)
        layout.addWidget(data_settings_group)
        
        # Save settings button
        save_settings_button = QPushButton("Save Settings")
        save_settings_button.clicked.connect(self.save_settings)
        layout.addWidget(save_settings_button)
        
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Settings")
    
    def setup_connections(self):
        """Setup signal connections"""
        self.connect_button.clicked.connect(self.connect_to_mt5)
        self.disconnect_button.clicked.connect(self.disconnect_from_mt5)
        
    def setup_monitoring(self):
        """Setup connection monitoring"""
        # Create and start monitoring thread
        self.monitor_thread = ConnectionMonitorThread(self.mt5_manager)
        self.monitor_thread.status_update.connect(self.update_monitoring_display)
        
    @pyqtSlot()
    def connect_to_mt5(self):
        """Connect to MetaTrader 5"""
        login = self.login_input.text()
        password = self.password_input.text()
        server = self.server_input.text()
        
        if not all([login, password, server]):
            self.log_message("Error: Please fill in all connection fields")
            return
        
        # Show progress
        self.connection_progress.setVisible(True)
        self.connection_progress.setRange(0, 0)  # Indeterminate progress
        self.connect_button.setEnabled(False)
        
        try:
            # Attempt connection
            success = self.mt5_manager.connect(login, password, server)
            
            if success:
                self.status_label.setText("Connected")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                self.connect_button.setEnabled(False)
                self.disconnect_button.setEnabled(True)
                
                # Start monitoring
                if not self.monitor_thread.isRunning():
                    self.monitor_thread.start()
                
                # Load symbols
                self.load_symbols()
                
                self.log_message(f"Successfully connected to {server}")
                self.connection_status_changed.emit(True)
                
            else:
                self.log_message("Failed to connect to MT5")
                self.status_label.setText("Connection Failed")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                
        except Exception as e:
            self.log_message(f"Connection error: {str(e)}")
            self.status_label.setText("Connection Error")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        finally:
            self.connection_progress.setVisible(False)
            self.connect_button.setEnabled(True)
    
    @pyqtSlot()
    def disconnect_from_mt5(self):
        """Disconnect from MetaTrader 5"""
        try:
            # Stop monitoring
            if self.monitor_thread and self.monitor_thread.isRunning():
                self.monitor_thread.stop()
                self.monitor_thread.wait()
            
            # Disconnect
            self.mt5_manager.disconnect()
            
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            
            # Clear symbols table
            self.symbols_table.setRowCount(0)
            
            self.log_message("Disconnected from MT5")
            self.connection_status_changed.emit(False)
            
        except Exception as e:
            self.log_message(f"Disconnection error: {str(e)}")
    
    def load_symbols(self):
        """Load available symbols into the table"""
        try:
            symbols = self.mt5_manager.get_symbols()
            
            self.symbols_table.setRowCount(len(symbols))
            
            for i, symbol in enumerate(symbols):
                self.symbols_table.setItem(i, 0, QTableWidgetItem(symbol))
                # TODO: Add real-time price data
                self.symbols_table.setItem(i, 1, QTableWidgetItem("0.00000"))
                self.symbols_table.setItem(i, 2, QTableWidgetItem("0.00000"))
                self.symbols_table.setItem(i, 3, QTableWidgetItem("0"))
                
        except Exception as e:
            self.log_message(f"Error loading symbols: {str(e)}")
    
    @pyqtSlot(dict)
    def update_monitoring_display(self, status):
        """Update the monitoring display with current status"""
        if 'account_info' in status:
            account_info = status['account_info']
            self.balance_label.setText(f"Balance: ${account_info.get('balance', 0):,.2f}")
            self.equity_label.setText(f"Equity: ${account_info.get('equity', 0):,.2f}")
            self.margin_label.setText(f"Margin: ${account_info.get('margin', 0):,.2f}")
            
            free_margin = account_info.get('equity', 0) - account_info.get('margin', 0)
            self.free_margin_label.setText(f"Free Margin: ${free_margin:,.2f}")
    
    def log_message(self, message):
        """Add a message to the log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def save_settings(self):
        """Save current settings"""
        # TODO: Implement settings persistence
        self.log_message("Settings saved successfully")
    
    def update_data(self, data):
        """Update widget with new MDPS data"""
        # This method allows the widget to receive updates from MDPS
        if 'symbols' in data:
            # Update symbol prices if available
            pass