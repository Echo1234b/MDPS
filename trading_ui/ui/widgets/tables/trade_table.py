from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTableWidget,
                           QTableWidgetItem, QHeaderView)

class TradeTable(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['Time', 'Symbol', 'Side', 'Price', 'Size', 'P&L'])
        
        # Make table fill the widget
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.table)

    def update_trades(self, trades):
        self.table.setRowCount(len(trades))
        for i, trade in enumerate(trades):
            self.table.setItem(i, 0, QTableWidgetItem(trade['time']))
            self.table.setItem(i, 1, QTableWidgetItem(trade['symbol']))
            self.table.setItem(i, 2, QTableWidgetItem(trade['side']))
            self.table.setItem(i, 3, QTableWidgetItem(str(trade['price'])))
            self.table.setItem(i, 4, QTableWidgetItem(str(trade['size'])))
            self.table.setItem(i, 5, QTableWidgetItem(str(trade['pnl'])))
