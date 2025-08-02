from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QTabWidget,
                           QWidget, QComboBox, QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # General settings tab
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        # Theme selection
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        general_layout.addLayout(theme_layout)

        # Update interval
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Update Interval (ms):")
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 10000)
        self.interval_spin.setValue(1000)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_spin)
        general_layout.addLayout(interval_layout)

        tab_widget.addTab(general_tab, "General")

        # Trading settings tab
        trading_tab = QWidget()
        trading_layout = QVBoxLayout(trading_tab)

        # Default quantity
        quantity_layout = QHBoxLayout()
        quantity_label = QLabel("Default Quantity:")
        self.quantity_input = QLineEdit()
        quantity_layout.addWidget(quantity_label)
        quantity_layout.addWidget(self.quantity_input)
        trading_layout.addLayout(quantity_layout)

        # Risk management
        self.risk_checkbox = QCheckBox("Enable Risk Management")
        trading_layout.addWidget(self.risk_checkbox)

        tab_widget.addTab(trading_tab, "Trading")

        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
