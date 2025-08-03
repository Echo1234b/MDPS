#!/usr/bin/env python3
"""
Enhanced MDPS Launcher
Multi-Dimensional Prediction System with Advanced PyQt UI

This is the main entry point for the enhanced MDPS system with full PyQt integration.
"""

import sys
import os
import traceback
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QFont
from PyQt5.QtCore import Qt, QTimer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mdps.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_splash_screen():
    """Create an attractive splash screen"""
    # Create a simple splash screen with MDPS branding
    pixmap = QPixmap(600, 400)
    pixmap.fill(Qt.white)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Background gradient
    from PyQt5.QtGui import QLinearGradient, QBrush
    gradient = QLinearGradient(0, 0, 600, 400)
    gradient.setColorAt(0, Qt.blue)
    gradient.setColorAt(1, Qt.darkBlue)
    painter.fillRect(pixmap.rect(), QBrush(gradient))
    
    # Title
    painter.setPen(Qt.white)
    title_font = QFont("Arial", 32, QFont.Bold)
    painter.setFont(title_font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter | Qt.AlignTop, "MDPS")
    
    # Subtitle
    subtitle_font = QFont("Arial", 16)
    painter.setFont(subtitle_font)
    painter.drawText(50, 150, 500, 50, Qt.AlignCenter, "Multi-Dimensional Prediction System")
    
    # Version and features
    features_font = QFont("Arial", 12)
    painter.setFont(features_font)
    features_text = [
        "✓ Advanced Machine Learning Models",
        "✓ Real-time Market Analysis", 
        "✓ Comprehensive Technical Analysis",
        "✓ Risk Management System",
        "✓ Strategy Optimization Engine",
        "✓ Live Performance Monitoring"
    ]
    
    y_pos = 200
    for feature in features_text:
        painter.drawText(50, y_pos, feature)
        y_pos += 25
    
    # Footer
    painter.drawText(pixmap.rect(), Qt.AlignCenter | Qt.AlignBottom, "Loading Enhanced Trading System...")
    
    painter.end()
    
    splash = QSplashScreen(pixmap)
    splash.setMask(pixmap.mask())
    return splash

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import pyqtgraph
    except ImportError:
        missing_deps.append("pyqtgraph")
    
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    if missing_deps:
        return False, missing_deps
    
    return True, []

def setup_application_style(app):
    """Setup application-wide styling"""
    app.setStyle('Fusion')
    
    # Set application stylesheet
    stylesheet = """
    QMainWindow {
        background-color: #f5f5f5;
    }
    
    QTabWidget::pane {
        border: 1px solid #c0c0c0;
        background-color: white;
    }
    
    QTabBar::tab {
        background-color: #e0e0e0;
        border: 1px solid #c0c0c0;
        padding: 8px 12px;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background-color: white;
        border-bottom: none;
    }
    
    QTabBar::tab:hover {
        background-color: #f0f0f0;
    }
    
    QGroupBox {
        font-weight: bold;
        border: 2px solid #cccccc;
        border-radius: 5px;
        margin-top: 1ex;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
    }
    
    QPushButton {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        font-size: 14px;
        margin: 4px 2px;
        border-radius: 4px;
    }
    
    QPushButton:hover {
        background-color: #45a049;
    }
    
    QPushButton:pressed {
        background-color: #3d8b40;
    }
    
    QPushButton:disabled {
        background-color: #cccccc;
        color: #666666;
    }
    
    QProgressBar {
        border: 2px solid grey;
        border-radius: 5px;
        text-align: center;
    }
    
    QProgressBar::chunk {
        background-color: #4CAF50;
        width: 20px;
    }
    
    QTableWidget {
        gridline-color: #d0d0d0;
        background-color: white;
        alternate-background-color: #f8f8f8;
    }
    
    QTableWidget::item {
        padding: 4px;
    }
    
    QTableWidget::item:selected {
        background-color: #3498db;
        color: white;
    }
    
    QListWidget::item {
        padding: 4px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    QListWidget::item:selected {
        background-color: #3498db;
        color: white;
    }
    """
    
    app.setStyleSheet(stylesheet)

def main():
    """Main application entry point"""
    try:
        logger.info("Starting MDPS Enhanced Trading System...")
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("MDPS - Multi-Dimensional Prediction System")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("MDPS Development Team")
        
        # Check dependencies
        deps_ok, missing_deps = check_dependencies()
        if not deps_ok:
            QMessageBox.critical(
                None, 
                "Missing Dependencies",
                f"The following required dependencies are missing:\n\n"
                f"{', '.join(missing_deps)}\n\n"
                f"Please install them using:\n"
                f"pip install {' '.join(missing_deps)}"
            )
            return 1
        
        # Set application icon
        icon_path = project_root / 'UI' / 'resources' / 'icons' / 'mdps.ico'
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
        
        # Setup application styling
        setup_application_style(app)
        
        # Create and show splash screen
        splash = create_splash_screen()
        splash.show()
        app.processEvents()
        
        # Initialize configuration
        splash.showMessage("Loading configuration...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()
        
        from config import MDPSConfig
        config = MDPSConfig()
        
        # Import and create main window
        splash.showMessage("Initializing user interface...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()
        
        from trading_ui.ui.main_window import MainWindow
        
        # Create main window
        splash.showMessage("Loading MDPS components...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()
        
        main_window = MainWindow(config)
        
        # Show main window and close splash
        splash.showMessage("Starting trading system...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()
        
        # Delay to show splash screen
        QTimer.singleShot(2000, lambda: (splash.close(), main_window.show()))
        
        logger.info("MDPS Enhanced Trading System started successfully")
        
        # Start the application event loop
        sys.exit(app.exec_())
        
    except ImportError as e:
        error_msg = f"Import Error: {str(e)}\n\nPlease ensure all required packages are installed:\n"
        error_msg += "pip install -r requirements.txt"
        
        if 'app' in locals():
            QMessageBox.critical(None, "Import Error", error_msg)
        else:
            print(error_msg)
        
        logger.error(f"Import error: {e}")
        return 1
        
    except Exception as e:
        error_msg = f"Error starting MDPS:\n{str(e)}\n\nTraceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
        
        if 'app' in locals():
            QMessageBox.critical(None, "System Error", error_msg)
        else:
            print(error_msg)
        
        logger.error(f"System error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)