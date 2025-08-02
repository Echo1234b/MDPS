from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer

def show_error_dialog(parent, title, message):
    """Show error dialog with given title and message"""
    QMessageBox.critical(parent, title, message)

def show_info_dialog(parent, title, message):
    """Show info dialog with given title and message"""
    QMessageBox.information(parent, title, message)

def create_update_timer(callback, interval=1000):
    """Create and start a timer for periodic updates"""
    timer = QTimer()
    timer.timeout.connect(callback)
    timer.start(interval)
    return timer

def format_timestamp(timestamp):
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")
