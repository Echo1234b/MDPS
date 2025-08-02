from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTableWidget,
                           QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt

class MarketTable(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Symbol', 'Bid', 'Ask', 'Volume', 'Change'])
        
        # Make table fill the widget
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.table)

    def update_market_data(self, data):
        self.table.setRowCount(len(data))
        for i, item in enumerate(data):
            self.table.setItem(i, 0, QTableWidgetItem(item['symbol']))
            self.table.setItem(i, 1, QTableWidgetItem(str(item['bid'])))
            self.table.setItem(i, 2, QTableWidgetItem(str(item['ask'])))
            self.table.setItem(i, 3, QTableWidgetItem(str(item['volume'])))
            self.table.setItem(i, 4, QTableWidgetItem(str(item['change'])))
