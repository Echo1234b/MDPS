"""
Position Sizer implementation.
"""

from typing import Dict, Any
import numpy as np
from ..exceptions import RiskManagementError

class PositionSizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sizing_strategy = config['position_sizing_method']
    
    def calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        Calculate optimal position size
        Args:
            signal: Trading signal
        Returns:
            float: Position size
        """
        try:
            if self.sizing_strategy == 'fixed_fractional':
                return self._fixed_fractional_sizing(signal)
            elif self.sizing_strategy == 'kelly_criterion':
                return self._kelly_sizing(signal)
            elif self.sizing_strategy == 'volatility_adjusted':
                return self._volatility_adjusted_sizing(signal)
            else:
                raise RiskManagementError(f"Unknown sizing strategy: {self.sizing_strategy}")
                
        except Exception as e:
            raise RiskManagementError(f"Position sizing failed: {str(e)}", {'strategy': self.sizing_strategy})
    
    def _fixed_fractional_sizing(self, signal: Dict[str, Any]) -> float:
        """
        Fixed fractional position sizing
        Args:
            signal: Trading signal
        Returns:
            float: Position size
        """
        account_size = signal.get('account_size', 0)
        risk_per_trade = self.config['risk_per_trade']
        stop_distance = signal.get('stop_distance', 0)
        
        if stop_distance == 0:
            return 0
        
        risk_amount = account_size * risk_per_trade
        position_size = risk_amount / stop_distance
        
        return position_size
    
    def _kelly_sizing(self, signal: Dict[str, Any]) -> float:
        """
        Kelly criterion position sizing
        Args:
            signal: Trading signal
        Returns:
            float: Position size
        """
        win_rate = signal.get('win_rate', 0.5)
        avg_win = signal.get('avg_win', 1)
        avg_loss = signal.get('avg_loss', 1)
        
        if avg_loss == 0:
            return 0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        account_size = signal.get('account_size', 0)
        return account_size * kelly_fraction
    
    def _volatility_adjusted_sizing(self, signal: Dict[str, Any]) -> float:
        """
        Volatility-adjusted position sizing
        Args:
            signal: Trading signal
        Returns:
            float: Position size
        """
        base_size = self._fixed_fractional_sizing(signal)
        volatility = signal.get('volatility', 1)
        target_volatility = signal.get('target_volatility', 1)
        
        if target_volatility == 0:
            return 0
        
        volatility_adjustment = target_volatility / volatility
        return base_size * volatility_adjustment
