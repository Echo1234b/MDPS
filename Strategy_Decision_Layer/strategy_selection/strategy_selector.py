"""
Strategy Selector implementation.
"""

from typing import Dict, Any, List
import numpy as np
from ..exceptions import StrategySelectionError

class StrategySelector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.available_strategies = {
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy()
        }
        self.performance_metrics = {}
        self.strategy_weights = config['strategy_weights']
    
    def select_strategy(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate trading strategy
        Args:
            signal: Trading signal
        Returns:
            dict: Selected strategy and parameters
        """
        try:
            market_conditions = self._analyze_market_conditions(signal)
            
            # Get strategy scores
            strategy_scores = {}
            for name, strategy in self.available_strategies.items():
                score = strategy.evaluate_market_fit(market_conditions)
                strategy_scores[name] = score
            
            # Apply weights
            final_scores = {
                name: score * self.strategy_weights.get(name, 1.0)
                for name, score in strategy_scores.items()
            }
            
            # Select best strategy
            best_strategy = max(final_scores.items(), key=lambda x: x[1])
            
            if best_strategy[1] < self.config['min_performance_threshold']:
                raise StrategySelectionError("No suitable strategy found", list(final_scores.keys()))
            
            return {
                'strategy': best_strategy[0],
                'score': best_strategy[1],
                'parameters': self.available_strategies[best_strategy[0]].get_parameters()
            }
            
        except Exception as e:
            raise StrategySelectionError(f"Strategy selection failed: {str(e)}", list(self.available_strategies.keys()))
    
    def _analyze_market_conditions(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market conditions
        Args:
            signal: Trading signal
        Returns:
            dict: Market condition analysis
        """
        return {
            'trend_strength': signal.get('trend_strength', 0),
            'volatility': signal.get('volatility', 0),
            'volume_profile': signal.get('volume_profile', {}),
            'market_regime': signal.get('market_regime', 'unknown')
        }
    
    def update_performance(self, strategy_name: str, performance: Dict[str, float]) -> None:
        """
        Update strategy performance metrics
        Args:
            strategy_name: Name of strategy
            performance: Performance metrics
        """
        if strategy_name not in self.performance_metrics:
            self.performance_metrics[strategy_name] = []
        
        self.performance_metrics[strategy_name].append(performance)
        
        # Update weights based on performance
        self._update_strategy_weights()
    
    def _update_strategy_weights(self) -> None:
        """
        Update strategy weights based on performance
        """
        for strategy_name in self.strategy_weights:
            if strategy_name in self.performance_metrics:
                recent_performance = self.performance_metrics[strategy_name][-10:]  # Last 10 trades
                if recent_performance:
                    avg_performance = np.mean([p['return'] for p in recent_performance])
                    # Adjust weight based on performance
                    adjustment = (avg_performance - 0.5) * 0.1  # 10% adjustment
                    self.strategy_weights[strategy_name] = max(0, min(1, 
                        self.strategy_weights[strategy_name] + adjustment))

class TrendFollowingStrategy:
    def evaluate_market_fit(self, conditions: Dict[str, Any]) -> float:
        """Evaluate how well market conditions suit trend following"""
        trend_strength = conditions.get('trend_strength', 0)
        volatility = conditions.get('volatility', 0)
        
        # Trend following works best with strong trends and moderate volatility
        score = trend_strength * 0.7 + (1 - abs(volatility - 0.5)) * 0.3
        return min(max(score, 0), 1)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'type': 'trend_following',
            'indicators': ['moving_average', 'adx'],
            'timeframes': ['daily', 'weekly']
        }

class MeanReversionStrategy:
    def evaluate_market_fit(self, conditions: Dict[str, Any]) -> float:
        """Evaluate how well market conditions suit mean reversion"""
        trend_strength = conditions.get('trend_strength', 0)
        volatility = conditions.get('volatility', 0)
        
        # Mean reversion works best with weak trends and high volatility
        score = (1 - trend_strength) * 0.7 + volatility * 0.3
        return min(max(score, 0), 1)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'type': 'mean_reversion',
            'indicators': ['rsi', 'bollinger'],
            'timeframes': ['hourly', 'daily']
        }

class BreakoutStrategy:
    def evaluate_market_fit(self, conditions: Dict[str, Any]) -> float:
        """Evaluate how well market conditions suit breakout trading"""
        volatility = conditions.get('volatility', 0)
        volume_profile = conditions.get('volume_profile', {})
        
        # Breakout works best with high volatility and strong volume
        volume_score = volume_profile.get('strength', 0)
        score = volatility * 0.6 + volume_score * 0.4
        return min(max(score, 0), 1)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return {
            'type': 'breakout',
            'indicators': ['volume', 'support_resistance'],
            'timeframes': ['15min', 'hourly']
        }
