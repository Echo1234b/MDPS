"""
Simulation Analysis Module
Handles trade simulation, backtesting, and post-trade analysis.
"""

from .trade_simulator import TradeSimulator
from .backtest_optimizer import BacktestOptimizer
from .post_trade_analyzer import PostTradeAnalyzer
from .feedback_loop import FeedbackLoop
from .slippage_simulator import SlippageSimulator
from .transaction_cost_modeler import TransactionCostModeler
from .execution_delay_emulator import ExecutionDelayEmulator

__all__ = [
    'TradeSimulator',
    'BacktestOptimizer',
    'PostTradeAnalyzer',
    'FeedbackLoop',
    'SlippageSimulator',
    'TransactionCostModeler',
    'ExecutionDelayEmulator'
]
