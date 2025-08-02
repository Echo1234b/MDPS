import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
import sys
from ui.main_window import MainWindow

@pytest.fixture
def app():
    """Create QApplication for testing"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()

def test_main_window_initialization(app):
    """Test main window initialization"""
    window = MainWindow()
    assert window.windowTitle() == 'Trading System'
    assert window.tab_widget.count() == 4  # Market, Technical, Trading, Analytics
    window.close()

def test_tab_switching(app):
    """Test tab switching functionality"""
    window = MainWindow()
    
    # Test switching to Technical tab
    QTest.mouseClick(window.tab_widget.tabBar(), Qt.LeftButton, pos=window.tab_widget.tabBar().tabRect(1).center())
    assert window.tab_widget.currentIndex() == 1
    
    # Test switching to Trading tab
    QTest.mouseClick(window.tab_widget.tabBar(), Qt.LeftButton, pos=window.tab_widget.tabBar().tabRect(2).center())
    assert window.tab_widget.currentIndex() == 2
    
    window.close()
