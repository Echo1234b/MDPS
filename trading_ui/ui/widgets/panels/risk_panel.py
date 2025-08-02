from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QProgressBar)

class RiskPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Risk Metrics")
        layout.addWidget(title)

        # Risk metrics
        self.margin_usage = self.create_metric("Margin Usage")
        self.drawdown = self.create_metric("Drawdown")
        self.exposure = self.create_metric("Exposure")

        layout.addWidget(self.margin_usage)
        layout.addWidget(self.drawdown)
        layout.addWidget(self.exposure)

    def create_metric(self, label_text):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel(label_text)
        progress = QProgressBar()
        
        layout.addWidget(label)
        layout.addWidget(progress)
        
        return widget

    def update_risk_metrics(self, metrics):
        self.margin_usage.findChild(QProgressBar).setValue(metrics['margin_usage'])
        self.drawdown.findChild(QProgressBar).setValue(metrics['drawdown'])
        self.exposure.findChild(QProgressBar).setValue(metrics['exposure'])
