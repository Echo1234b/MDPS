"""
Stop/Target Generator implementation.
"""

from typing import Dict, Any, Tuple
import numpy as np
from ..exceptions import RiskManagementError

class StopTargetGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generation_rules = {
            'min_risk_reward_ratio': 2.0,
            'atr_multiplier': 2.0,
            'use_structure': True
        }
    
    def generate_levels(self, signal: Dict[str, Any]) -> Tuple[float, float]:
        """
        Generate stop loss and take profit levels
        Args:
            signal: Trading signal
        Returns:
            tuple: (stop_loss, take_profit)
        """
        try:
            entry_price = signal.get('entry_price', 0)
            direction = signal.get('direction')
            
            if not entry_price or not direction:
                raise RiskManagementError("Invalid signal parameters")
            
            # Generate stop loss level
            stop_loss = self._generate_stop_level(signal)
            
            # Generate take profit level
            take_profit = self._generate_target_level(signal, stop_loss)
            
            return stop_loss, take_profit
            
        except Exception as e:
            raise RiskManagementError(f"Stop/target generation failed: {str(e)}", signal)
    
    def _generate_stop_level(self, signal: Dict[str, Any]) -> float:
        """
        Generate stop loss level
        Args:
            signal: Trading signal
        Returns:
            float: Stop loss level
        """
        entry_price = signal.get('entry_price')
        direction = signal.get('direction')
        atr = signal.get('atr', 0)
        
        if self.generation_rules['use_structure']:
            # Use market structure for stop placement
            structure_stop = self._get_structure_stop(signal)
            if structure_stop:
                return structure_stop
        
        # Use ATR-based stop
        atr_distance = atr * self.generation_rules['atr_multiplier']
        
        if direction == 'long':
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance
    
    def _generate_target_level(self, signal: Dict[str, Any], stop_loss: float) -> float:
        """
        Generate take profit level
        Args:
            signal: Trading signal
            stop_loss: Stop loss level
        Returns:
            float: Take profit level
        """
        entry_price = signal.get('entry_price')
        direction = signal.get('direction')
        
        # Calculate risk amount
        if direction == 'long':
            risk_amount = entry_price - stop_loss
        else:
            risk_amount = stop_loss - entry_price
        
        # Apply minimum risk/reward ratio
        reward_amount = risk_amount * self.generation_rules['min_risk_reward_ratio']
        
        if direction == 'long':
            return entry_price + reward_amount
        else:
            return entry_price - reward_amount
    
    def _get_structure_stop(self, signal: Dict[str, Any]) -> float:
        """
        Get stop level based on market structure
        Args:
            signal: Trading signal
        Returns:
            float: Structure-based stop level or None
        """
        direction = signal.get('direction')
        structure_levels = signal.get('structure_levels', {})
        
        if direction == 'long':
            return structure_levels.get('support')
        else:
            return structure_levels.get('resistance')
