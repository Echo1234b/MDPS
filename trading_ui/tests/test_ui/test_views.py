import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
import sys
from ui.views.market_view import MarketView
from ui.views.technical_view import TechnicalView
from ui.views.trading_view import TradingView
from ui.views.analytics_view import AnalyticsView

@pytest.fixture
def app():
    """Create QApplication for testing"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()

def test_market_view_initialization(app):
    """Test market view initialization"""
    view = MarketView(None)
    assert view.price_chart is not None
    assert view.orderbook_chart is not None
    assert view.volume_profile is not None
    assert view.market_table is not None

def test_technical_view_initialization(app):
    """Test technical view initialization"""
    view = TechnicalView(None)
    assert view.indicator_selector.count() > 0
    assert view.timeframe_selector.count() > 0
    assert view.chart is not None

def test_trading_view_initialization(app):
    """Test trading view initialization"""
    view = TradingView(None)
    assert view.symbol_input is not None
    assert view.quantity_input is not None
    assert view.buy_button is not None
    assert view.sell_button is not None

def test_analytics_view_initialization(app):
    """Test analytics view initialization"""
    view = AnalyticsView(None)
    assert view.accuracy_label is not None
    assert view.accuracy_bar is not None
    assert view.confidence_label is not None
    assert view.confidence_bar is not None
