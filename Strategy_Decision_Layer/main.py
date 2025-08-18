"""
Main implementation of the Strategy & Decision Layer.
"""

from .config import Config
from .signal_validation.signal_validator import SignalValidator
from .risk_management.risk_manager import RiskManager
from .strategy_selection.strategy_selector import StrategySelector
from .timing_execution.timing_optimizer import TimingOptimizer
from .simulation_analysis.trade_simulator import TradeSimulator
from .exceptions import StrategyDecisionError

class StrategyDecisionLayer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.signal_validator = SignalValidator(self.config.SIGNAL_VALIDATION)
        self.risk_manager = RiskManager(self.config.RISK_MANAGEMENT)
        self.strategy_selector = StrategySelector(self.config.STRATEGY_SELECTION)
        self.timing_optimizer = TimingOptimizer(self.config.TIMING_EXECUTION)
        self.trade_simulator = TradeSimulator(self.config.SIMULATION)

    def process_signal(self, signal):
        """
        Main signal processing pipeline
        Args:
            signal: Trading signal to be processed
        Returns:
            Executed trade or None if signal is rejected
        """
        try:
            # Signal Validation
            validated_signal = self.signal_validator.validate_signal(signal)
            if not validated_signal:
                return None

            # Risk Assessment
            risk_assessed = self.risk_manager.assess_risk(validated_signal)
            if not risk_assessed:
                return None

            # Strategy Selection
            strategy = self.strategy_selector.select_strategy(validated_signal)
            if not strategy:
                return None

            # Timing Optimization
            timing = self.timing_optimizer.optimize_entry(validated_signal)
            if not timing:
                return None

            # Execute Trade
            return self.execute_trade(validated_signal, strategy, timing)

        except Exception as e:
            raise StrategyDecisionError(f"Error processing signal: {str(e)}")

    def execute_trade(self, signal, strategy, timing):
        """
        Execute trade with given parameters
        Args:
            signal: Validated trading signal
            strategy: Selected trading strategy
            timing: Optimized execution timing
        Returns:
            Executed trade details
        """
        try:
            # Position sizing
            position_size = self.risk_manager.calculate_position_size(signal)
            
            # Stop loss and take profit levels
            stop_loss, take_profit = self.risk_manager.generate_stops_targets(signal)
            
            # Execute trade
            trade = {
                'signal': signal,
                'strategy': strategy,
                'timing': timing,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            return trade

        except Exception as e:
            raise StrategyDecisionError(f"Error executing trade: {str(e)}")
