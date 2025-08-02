from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from pyqtgraph import PlotWidget
import numpy as np

class VolumeProfile(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

    def update_volume_profile(self, prices, volumes):
        self.plot_widget.clear()
        if prices is not None and volumes is not None:
            self.plot_widget.plot(volumes, prices, pen='b')
