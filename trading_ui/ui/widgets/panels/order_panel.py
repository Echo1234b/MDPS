from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QTableWidget, QTableWidgetItem)

class OrderPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Orders")
        layout.addWidget(title)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Order ID', 'Symbol', 'Type', 'Size', 'Status'])
        layout.addWidget(self.table)

    def update_orders(self, orders):
        self.table.setRowCount(len(orders))
        for i, order in enumerate(orders):
            self.table.setItem(i, 0, QTableWidgetItem(order['id']))
            self.table.setItem(i, 1, QTableWidgetItem(order['symbol']))
            self.table.setItem(i, 2, QTableWidgetItem(order['type']))
            self.table.setItem(i, 3, QTableWidgetItem(str(order['size'])))
            self.table.setItem(i, 4, QTableWidgetItem(order['status']))
