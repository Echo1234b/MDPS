"""
Risk Manager implementation.
"""

from typing import Dict, Any, List
import numpy as np
from ..exceptions import RiskManagementError

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_limits = {
            'max_portfolio_risk': config['max_portfolio_risk'],
            'max_drawdown': config['max_drawdown'],
            'risk_per_trade': config['risk_per_trade'],
            'max_correlation': config['max_correlation']
        }
        self.position_sizer = PositionSizer(config)
        self.stop_target_generator = StopTargetGenerator(config)
    
    def assess_risk(self, signal: Dict[str, Any]) -> bool:
        """
        Assess overall risk of proposed trade
        Args:
            signal: Trading signal
        Returns:
            bool: True if risk is acceptable
        """
        try:
            # Check portfolio-level risk
            if not self._check_portfolio_risk(signal):
                return False
            
            # Check drawdown limits
            if not self._check_drawdown_limits(signal):
                return False
            
            # Check correlation risk
            if not self._check_correlation_risk(signal):
                return False
            
            return True
            
        except Exception as e:
            raise RiskManagementError(f"Risk assessment failed: {str(e)}", self.risk_limits)
    
    def _check_portfolio_risk(self, signal: Dict[str, Any]) -> bool:
        """
        Check portfolio-level risk limits
        Args:
            signal: Trading signal
        Returns:
            bool: True if within limits
        """
        current_risk = signal.get('portfolio_risk', 0)
        proposed_risk = signal.get('trade_risk', 0)
        
        total_risk = current_risk + proposed_risk
        return total_risk <= self.risk_limits['max_portfolio_risk']
    
    def _check_drawdown_limits(self, signal: Dict[str, Any]) -> bool:
        """
        Check drawdown limits
        Args:
            signal: Trading signal
        Returns:
            bool: True if within limits
        """
        current_drawdown = signal.get('current_drawdown', 0)
        return current_drawdown <= self.risk_limits['max_drawdown']
    
    def _check_correlation_risk(self, signal: Dict[str, Any]) -> bool:
        """
        Check correlation risk with existing positions
        Args:
            signal: Trading signal
        Returns:
            bool: True if within limits
        """
        correlations = signal.get('asset_correlations', [])
        if not correlations:
            return True
        
        max_correlation = max(abs(c) for c in correlations)
        return max_correlation <= self.risk_limits['max_correlation']
    
    def calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        Calculate position size for trade
        Args:
            signal: Trading signal
        Returns:
            float: Position size
        """
        return self.position_sizer.calculate_position_size(signal)
    
    def generate_stops_targets(self, signal: Dict[str, Any]) -> tuple:
        """
        Generate stop loss and take profit levels
        Args:
            signal: Trading signal
        Returns:
            tuple: (stop_loss, take_profit)
        """
        return self.stop_target_generator.generate_levels(signal)
