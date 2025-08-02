from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtChart import QChart, QChartView, QCandlestickSeries, QCandlestickSet
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QDateTime
from pyqtgraph import PlotWidget
import numpy as np

class PriceChart(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

    def update_data(self, data):
        self.plot_widget.clear()
        if data is not None and len(data) > 0:
            self.plot_widget.plot(data['close'])
