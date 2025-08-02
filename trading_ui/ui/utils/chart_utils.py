import numpy as np
from PyQt5.QtGui import QColor

def format_price(value, decimals=2):
    """Format price values with appropriate decimal places"""
    return f"{value:.{decimals}f}"

def format_volume(value):
    """Format volume values with appropriate suffixes"""
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    return str(value)

def get_color(value, reference=0):
    """Get color based on value comparison to reference"""
    if value > reference:
        return QColor(0, 255, 0)  # Green
    elif value < reference:
        return QColor(255, 0, 0)  # Red
    return QColor(255, 255, 255)  # White

def calculate_moving_average(data, window):
    """Calculate simple moving average"""
    return np.convolve(data, np.ones(window)/window, mode='valid')
