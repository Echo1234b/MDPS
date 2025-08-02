"""
Trading UI Views
"""
from .base_view import BaseView
from .market_view import MarketView
from .technical_view import TechnicalView
from .trading_view import TradingView
from .analytics_view import AnalyticsView
from .system_monitor_view import SystemMonitorView
from .knowledge_graph_view import KnowledgeGraphView
from .external_factors_view import ExternalFactorsView
from .market_structure_view import MarketStructureView
from .pattern_recognition_view import PatternRecognitionView
from .model_comparison_view import ModelComparisonView
from .strategy_simulator_view import StrategySimulatorView
from .risk_management_view import RiskManagementView
from .data_quality_view import DataQualityView

__all__ = [
    'BaseView',
    'MarketView',
    'TechnicalView',
    'TradingView',
    'AnalyticsView',
    'SystemMonitorView',
    'KnowledgeGraphView',
    'ExternalFactorsView',
    'MarketStructureView',
    'PatternRecognitionView',
    'ModelComparisonView',
    'StrategySimulatorView',
    'RiskManagementView',
    'DataQualityView'
]
