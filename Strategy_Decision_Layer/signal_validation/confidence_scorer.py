"""
Confidence Scorer implementation.
"""

from typing import Dict, Any, List
import numpy as np
from ..exceptions import SignalValidationError

class ConfidenceScorer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scoring_weights = {
            'model_confidence': 0.4,
            'historical_performance': 0.3,
            'market_regime': 0.2,
            'volume_confirmation': 0.1
        }
    
    def calculate_confidence(self, signal: Dict[str, Any]) -> float:
        """
        Calculate confidence score for signal
        Args:
            signal: Trading signal
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            scores = {}
            
            # Model confidence score
            scores['model_confidence'] = signal.get('model_confidence', 0)
            
            # Historical performance score
            scores['historical_performance'] = self._calculate_historical_score(signal)
            
            # Market regime score
            scores['market_regime'] = self._calculate_regime_score(signal)
            
            # Volume confirmation score
            scores['volume_confirmation'] = self._calculate_volume_score(signal)
            
            # Calculate weighted average
            total_score = sum(
                score * weight 
                for score, weight in zip(scores.values(), self.scoring_weights.values())
            )
            
            return min(max(total_score, 0), 1)  # Clamp between 0 and 1
            
        except Exception as e:
            raise SignalValidationError(f"Confidence calculation failed: {str(e)}", signal)
    
    def _calculate_historical_score(self, signal: Dict[str, Any]) -> float:
        """
        Calculate historical performance score
        Args:
            signal: Trading signal
        Returns:
            float: Historical performance score
        """
        historical_data = signal.get('historical_performance', [])
        if not historical_data:
            return 0.5  # Neutral score if no historical data
        
        # Calculate win rate from historical data
        wins = sum(1 for trade in historical_data if trade['pnl'] > 0)
        total = len(historical_data)
        
        return wins / total if total > 0 else 0.5
    
    def _calculate_regime_score(self, signal: Dict[str, Any]) -> float:
        """
        Calculate market regime score
        Args:
            signal: Trading signal
        Returns:
            float: Market regime score
        """
        regime = signal.get('market_regime', 'unknown')
        regime_scores = {
            'trending': 0.8,
            'ranging': 0.6,
            'volatile': 0.4,
            'unknown': 0.5
        }
        
        return regime_scores.get(regime, 0.5)
    
    def _calculate_volume_score(self, signal: Dict[str, Any]) -> float:
        """
        Calculate volume confirmation score
        Args:
            signal: Trading signal
        Returns:
            float: Volume confirmation score
        """
        current_volume = signal.get('current_volume', 0)
        average_volume = signal.get('average_volume', 1)
        
        if average_volume == 0:
            return 0.5
        
        volume_ratio = current_volume / average_volume
        
        # Normalize volume ratio to 0-1 range
        return min(volume_ratio / 2, 1)  # Cap at 1 (2x average volume)
    
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update scoring weights based on performance
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # Implement weight update logic based on performance
        for metric, value in performance_metrics.items():
            if metric in self.scoring_weights:
                # Adjust weights based on performance
                adjustment = (value - 0.5) * 0.1  # 10% adjustment
                self.scoring_weights[metric] = max(0, min(1, 
                    self.scoring_weights[metric] + adjustment))
