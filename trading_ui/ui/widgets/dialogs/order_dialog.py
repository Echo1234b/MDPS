from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                           QLabel, QLineEdit, QComboBox, QPushButton,
                           QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt

class OrderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Place Order")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Symbol
        symbol_layout = QHBoxLayout()
        symbol_label = QLabel("Symbol:")
        self.symbol_input = QLineEdit()
        symbol_layout.addWidget(symbol_label)
        symbol_layout.addWidget(self.symbol_input)
        layout.addLayout(symbol_layout)

        # Order type
        type_layout = QHBoxLayout()
        type_label = QLabel("Order Type:")
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Market", "Limit", "Stop"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Side
        side_layout = QHBoxLayout()
        side_label = QLabel("Side:")
        self.side_combo = QComboBox()
        self.side_combo.addItems(["Buy", "Sell"])
        side_layout.addWidget(side_label)
        side_layout.addWidget(self.side_combo)
        layout.addLayout(side_layout)

        # Quantity
        quantity_layout = QHBoxLayout()
        quantity_label = QLabel("Quantity:")
        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(1, 1000000)
        quantity_layout.addWidget(quantity_label)
        quantity_layout.addWidget(self.quantity_input)
        layout.addLayout(quantity_layout)

        # Price (for limit/stop orders)
        price_layout = QHBoxLayout()
        price_label = QLabel("Price:")
        self.price_input = QDoubleSpinBox()
        self.price_input.setRange(0, 1000000)
        self.price_input.setDecimals(2)
        price_layout.addWidget(price_label)
        price_layout.addWidget(self.price_input)
        layout.addLayout(price_layout)

        # Buttons
        button_layout = QHBoxLayout()
        submit_button = QPushButton("Submit")
        cancel_button = QPushButton("Cancel")
        
        submit_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(submit_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Connect signals
        self.type_combo.currentTextChanged.connect(self.on_type_changed)

    def on_type_changed(self, order_type):
        # Enable/disable price input based on order type
        self.price_input.setEnabled(order_type != "Market")

    def get_order_data(self):
        return {
            'symbol': self.symbol_input.text(),
            'type': self.type_combo.currentText(),
            'side': self.side_combo.currentText(),
            'quantity': self.quantity_input.value(),
            'price': self.price_input.value() if self.price_input.isEnabled() else None
        }
