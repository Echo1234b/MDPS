from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from pyqtgraph import PlotWidget
import numpy as np

class OrderBookChart(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

    def update_orderbook(self, bids, asks):
        self.plot_widget.clear()
        if bids is not None and asks is not None:
            # Plot bids
            self.plot_widget.plot(bids[:, 0], bids[:, 1], pen='g')
            # Plot asks
            self.plot_widget.plot(asks[:, 0], asks[:, 1], pen='r')
