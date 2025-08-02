from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QTableWidget, QTableWidgetItem)

class PositionPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Positions")
        layout.addWidget(title)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Symbol', 'Size', 'Entry', 'Current', 'P&L'])
        layout.addWidget(self.table)

    def update_position(self, positions):
        self.table.setRowCount(len(positions))
        for i, pos in enumerate(positions):
            self.table.setItem(i, 0, QTableWidgetItem(pos['symbol']))
            self.table.setItem(i, 1, QTableWidgetItem(str(pos['size'])))
            self.table.setItem(i, 2, QTableWidgetItem(str(pos['entry'])))
            self.table.setItem(i, 3, QTableWidgetItem(str(pos['current'])))
            self.table.setItem(i, 4, QTableWidgetItem(str(pos['pnl'])))
