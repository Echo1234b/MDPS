"""
Direction Filter implementation.
"""

from typing import Dict, Any, Optional
import numpy as np
from ..exceptions import SignalValidationError

class DirectionFilter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.filter_rules = {
            'min_trend_strength': 0.3,
            'require_volume_confirmation': True,
            'min_momentum': 0.2
        }
    
    def validate_direction(self, signal: Dict[str, Any], market_context: Dict[str, Any]) -> bool:
        """
        Validate trade direction
        Args:
            signal: Trading signal
            market_context: Current market context
        Returns:
            bool: True if direction is valid
        """
        try:
            signal_direction = signal.get('direction')
            if not signal_direction:
                return False
            
            # Check trend alignment
            if not self._check_trend_alignment(signal_direction, market_context):
                return False
            
            # Check momentum confirmation
            if not self._check_momentum_confirmation(signal_direction, market_context):
                return False
            
            # Check volume confirmation if required
            if (self.filter_rules['require_volume_confirmation'] and 
                not self._check_volume_confirmation(signal_direction, market_context)):
                return False
            
            return True
            
        except Exception as e:
            raise SignalValidationError(f"Direction validation failed: {str(e)}", signal)
    
    def _check_trend_alignment(self, direction: str, market_context: Dict[str, Any]) -> bool:
        """
        Check alignment with market trend
        Args:
            direction: Signal direction
            market_context: Market context data
        Returns:
            bool: True if aligned with trend
        """
        trend_strength = market_context.get('trend_strength', 0)
        if trend_strength < self.filter_rules['min_trend_strength']:
            return True  # Weak trend, allow both directions
        
        market_trend = market_context.get('trend_direction')
        if not market_trend:
            return True
        
        return direction == market_trend
    
    def _check_momentum_confirmation(self, direction: str, market_context: Dict[str, Any]) -> bool:
        """
        Check momentum confirmation
        Args:
            direction: Signal direction
            market_context: Market context data
        Returns:
            bool: True if momentum confirms direction
        """
        momentum = market_context.get('momentum', 0)
        if abs(momentum) < self.filter_rules['min_momentum']:
            return True  # Weak momentum, allow both directions
        
        return (direction == 'long' and momentum > 0) or (direction == 'short' and momentum < 0)
    
    def _check_volume_confirmation(self, direction: str, market_context: Dict[str, Any]) -> bool:
        """
        Check volume confirmation
        Args:
            direction: Signal direction
            market_context: Market context data
        Returns:
            bool: True if volume confirms direction
        """
        volume_profile = market_context.get('volume_profile', {})
        if not volume_profile:
            return True
        
        buy_volume = volume_profile.get('buy_volume', 0)
        sell_volume = volume_profile.get('sell_volume', 0)
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return True
        
        volume_ratio = (buy_volume / total_volume if direction == 'long' 
                       else sell_volume / total_volume)
        
        return volume_ratio > 0.6  # Require 60% volume confirmation
