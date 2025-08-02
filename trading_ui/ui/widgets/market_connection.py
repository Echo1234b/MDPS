"""Market data feed view for MT5 connection and data management"""
import sys
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QComboBox, QGridLayout, QGroupBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# Add project root to Python path
root_path = str(Path(__file__).parent.parent.parent.parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from Data_Collection_and_Acquisition import MT5ConnectionManager

class MarketConnectionWidget(QWidget):
    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.mt5_manager = MT5ConnectionManager(config=self.config)
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
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
        
        if self.config and self.config.mt5_settings:
            self.server_input.setText(self.config.mt5_settings.get("server", ""))
            
        self.connect_button = QPushButton("Connect to MT5")
        self.status_label = QLabel("Not Connected")
        self.status_label.setStyleSheet("color: red")
        
        # Add widgets to layout
        connection_layout.addWidget(QLabel("Login:"), 0, 0)
        connection_layout.addWidget(self.login_input, 0, 1)
        connection_layout.addWidget(QLabel("Password:"), 1, 0)
        connection_layout.addWidget(self.password_input, 1, 1)
        connection_layout.addWidget(QLabel("Server:"), 2, 0)
        connection_layout.addWidget(self.server_input, 2, 1)
        connection_layout.addWidget(QLabel("Status:"), 3, 0)
        connection_layout.addWidget(self.status_label, 3, 1)
        connection_layout.addWidget(self.connect_button, 3, 2)
        
        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)
        
        # Market Data group
        market_group = QGroupBox("Market Data")
        market_layout = QGridLayout()
        
        # Symbol selection
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEnabled(False)
        
        # Timeframe selection
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"])
        self.timeframe_combo.setEnabled(False)
        
        # Add to layout
        market_layout.addWidget(QLabel("Symbol:"), 0, 0)
        market_layout.addWidget(self.symbol_combo, 0, 1)
        market_layout.addWidget(QLabel("Timeframe:"), 1, 0)
        market_layout.addWidget(self.timeframe_combo, 1, 1)
        
        market_group.setLayout(market_layout)
        layout.addWidget(market_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
    def setup_connections(self):
        """Setup signal/slot connections"""
        self.connect_button.clicked.connect(self.toggle_connection)
        self.mt5_manager.connection_success.connect(self.on_connection_success)
        self.mt5_manager.connection_error.connect(self.on_connection_error)
        self.mt5_manager.connection_status.connect(self.on_connection_status_changed)
        
    def toggle_connection(self):
        """Handle connection button click"""
        if not self.mt5_manager.is_connected:
            self.connect_button.setEnabled(False)
            self.status_label.setText("Connecting...")
            self.status_label.setStyleSheet("color: orange")
            
            # Get credentials
            login = self.login_input.text().strip()
            password = self.password_input.text().strip()
            server = self.server_input.text().strip()
            
            # Validate inputs
            if not login or not password or not server:
                self.status_label.setText("Please fill in all fields")
                self.status_label.setStyleSheet("color: red")
                self.connect_button.setEnabled(True)
                return
                
            try:
                # Convert login to integer and connect
                login = int(login)
                self.mt5_manager.connect(login=login, password=password, server=server)
            except ValueError:
                self.status_label.setText("Login ID must be a number")
                self.status_label.setStyleSheet("color: red")
                self.connect_button.setEnabled(True)
            except Exception as e:
                self.status_label.setText(f"Connection error: {str(e)}")
                self.status_label.setStyleSheet("color: red")
                self.connect_button.setEnabled(True)
        else:
            # Handle disconnection
            self.connect_button.setEnabled(False)
            if self.mt5_manager.disconnect():
                self.connect_button.setText("Connect to MT5")
                self.status_label.setText("Disconnected")
                self.status_label.setStyleSheet("color: orange")
                self.symbol_combo.setEnabled(False)
                self.timeframe_combo.setEnabled(False)
            self.connect_button.setEnabled(True)
            
    def on_connection_success(self, info):
        """Handle successful connection"""
        self.connect_button.setEnabled(True)
        self.connect_button.setText("Disconnect")
        self.status_label.setText("Connected")
        self.status_label.setStyleSheet("color: green")
        
        # Enable and populate symbol selection
        symbols = self.mt5_manager.get_symbols()
        self.symbol_combo.clear()
        self.symbol_combo.addItems([s.name for s in symbols])
        self.symbol_combo.setEnabled(True)
        self.timeframe_combo.setEnabled(True)
        
    def on_connection_error(self, error):
        """Handle connection error"""
        self.connect_button.setEnabled(True)
        self.status_label.setText("Connection Failed")
        self.status_label.setStyleSheet("color: red")
        
    def on_connection_status_changed(self, connected):
        """Handle connection status changes"""
        if not connected and self.mt5_manager.is_connected:
            self.connect_button.setText("Connect to MT5")
            self.status_label.setText("Connection Lost")
            self.status_label.setStyleSheet("color: red")
            self.symbol_combo.setEnabled(False)
            self.timeframe_combo.setEnabled(False)
