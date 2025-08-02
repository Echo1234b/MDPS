"""
Slippage Simulator implementation.
"""

from typing import Dict, Any, List
import numpy as np
from ..exceptions import SimulationError

class SlippageSimulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.slippage_model = {
            'base_rate': config['slippage_rate'],
            'size_impact': 0.0001,  # Impact per unit size
            'volatility_factor': 0.5,
            'liquidity_factor': 0.3
        }
    
    def simulate_slippage(self, order: Dict[str, Any], market_conditions: Dict[str, Any]) -> float:
        """
        Simulate order slippage
        Args:
            order: Order details
            market_conditions: Current market conditions
        Returns:
            float: Simulated slippage
        """
        try:
            base_slippage = self.slippage_model['base_rate']
            
            # Calculate size impact
            size_impact = self._calculate_size_impact(order)
            
            # Calculate volatility impact
            volatility_impact = self._calculate_volatility_impact(market_conditions)
            
            # Calculate liquidity impact
            liquidity_impact = self._calculate_liquidity_impact(market_conditions)
            
            # Combine all factors
            total_slippage = (
                base_slippage +
                size_impact +
                volatility_impact +
                liquidity_impact
            )
            
            # Apply random noise
            noise = np.random.normal(0, 0.0001)
            total_slippage += noise
            
            return total_slippage
            
        except Exception as e:
            raise SimulationError(f"Slippage simulation failed: {str(e)}", {'order': order, 'market': market_conditions})
    
    def analyze_slippage(self, slippage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze slippage patterns
        Args:
            slippage_data: List of slippage records
        Returns:
            dict: Slippage analysis
        """
        try:
            if not slippage_data:
                return {}
            
            slippages = [d['slippage'] for d in slippage_data]
            
            return {
                'avg_slippage': np.mean(slippages),
                'std_slippage': np.std(slippages),
                'max_slippage': max(slippages),
                'min_slippage': min(slippages),
                'slippage_distribution': self._calculate_distribution(slippages),
                'time_analysis': self._analyze_time_patterns(slippage_data)
            }
            
        except Exception as e:
            raise SimulationError(f"Slippage analysis failed: {str(e)}", {'data_points': len(slippage_data)})
    
    def _calculate_size_impact(self, order: Dict[str, Any]) -> float:
        """
        Calculate slippage impact due to order size
        Args:
            order: Order details
        Returns:
            float: Size impact on slippage
        """
        order_size = order.get('size', 0)
        market_size = order.get('market_size', 1)
        
        if market_size == 0:
            return 0
        
        size_ratio = order_size / market_size
        return size_ratio * self.slippage_model['size_impact']
    
    def _calculate_volatility_impact(self, market_conditions: Dict[str, Any]) -> float:
        """
        Calculate slippage impact due to market volatility
        Args:
            market_conditions: Market conditions
        Returns:
            float: Volatility impact on slippage
        """
        volatility = market_conditions.get('volatility', 0)
        return volatility * self.slippage_model['volatility_factor']
    
    def _calculate_liquidity_impact(self, market_conditions: Dict[str, Any]) -> float:
        """
        Calculate slippage impact due to market liquidity
        Args:
            market_conditions: Market conditions
        Returns:
            float: Liquidity impact on slippage
        """
        liquidity = market_conditions.get('liquidity', 1)
        if liquidity == 0:
            return 0
        
        return (1 / liquidity) * self.slippage_model['liquidity_factor']
    
    def _calculate_distribution(self, slippages: List[float]) -> Dict[str, float]:
        """
        Calculate slippage distribution statistics
        Args:
            slippages: List of slippage values
        Returns:
            dict: Distribution statistics
        """
        return {
            'percentile_10': np.percentile(slippages, 10),
            'percentile_25': np.percentile(slippages, 25),
            'percentile_50': np.median(slippages),
            'percentile_75': np.percentile(slippages, 75),
            'percentile_90': np.percentile(slippages, 90)
        }
    
    def _analyze_time_patterns(self, slippage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze time-based patterns in slippage
        Args:
            slippage_data: List of slippage records with timestamps
        Returns:
            dict: Time pattern analysis
        """
        # Group slippage by time periods
        hourly_slippage = {}
        for record in slippage_data:
            hour = record['timestamp'].hour
            if hour not in hourly_slippage:
                hourly_slippage[hour] = []
            hourly_slippage[hour].append(record['slippage'])
        
        # Calculate hourly averages
        hourly_avg = {
            hour: np.mean(slippages)
            for hour, slippages in hourly_slippage.items()
        }
        
        return {
            'hourly_averages': hourly_avg,
            'best_hour': min(hourly_avg.items(), key=lambda x: x[1])[0],
            'worst_hour': max(hourly_avg.items(), key=lambda x: x[1])[0]
        }
