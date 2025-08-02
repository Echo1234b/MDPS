"""
Execution Delay Emulator implementation.
"""

from typing import Dict, Any, List
import numpy as np
from ..exceptions import SimulationError

class ExecutionDelayEmulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.delay_model = {
            'base_delay': config['execution_delay_threshold'],
            'size_factor': 0.1,
            'volatility_factor': 0.2,
            'liquidity_factor': 0.3,
            'time_of_day_factors': {
                'open': 1.5,
                'close': 1.5,
                'normal': 1.0
            }
        }
    
    def emulate_delay(self, order: Dict[str, Any], market_conditions: Dict[str, Any]) -> float:
        """
        Emulate execution delay for an order
        Args:
            order: Order details
            market_conditions: Current market conditions
        Returns:
            float: Emulated delay in milliseconds
        """
        try:
            base_delay = self.delay_model['base_delay']
            
            # Calculate size impact
            size_impact = self._calculate_size_impact(order)
            
            # Calculate volatility impact
            volatility_impact = self._calculate_volatility_impact(market_conditions)
            
            # Calculate liquidity impact
            liquidity_impact = self._calculate_liquidity_impact(market_conditions)
            
            # Calculate time of day impact
            time_impact = self._calculate_time_impact(order)
            
            # Combine all factors
            total_delay = (
                base_delay +
                size_impact +
                volatility_impact +
                liquidity_impact
            ) * time_impact
            
            # Add random noise
            noise = np.random.normal(0, base_delay * 0.1)
            total_delay += noise
            
            return max(0, total_delay)  # Ensure non-negative delay
            
        except Exception as e:
            raise SimulationError(f"Delay emulation failed: {str(e)}", {'order': order, 'market': market_conditions})
    
    def analyze_impact(self, delay_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze delay impact on trading performance
        Args:
            delay_data: List of delay records
        Returns:
            dict: Delay impact analysis
        """
        try:
            if not delay_data:
                return {}
            
            delays = [d['delay'] for d in delay_data]
            price_impacts = [d.get('price_impact', 0) for d in delay_data]
            
            return {
                'avg_delay': np.mean(delays),
                'std_delay': np.std(delays),
                'max_delay': max(delays),
                'min_delay': min(delays),
                'avg_price_impact': np.mean(price_impacts),
                'delay_distribution': self._calculate_distribution(delays),
                'time_analysis': self._analyze_time_patterns(delay_data),
                'cost_analysis': self._analyze_delay_costs(delay_data)
            }
            
        except Exception as e:
            raise SimulationError(f"Delay analysis failed: {str(e)}", {'data_points': len(delay_data)})
    
    def _calculate_size_impact(self, order: Dict[str, Any]) -> float:
        """
        Calculate delay impact due to order size
        Args:
            order: Order details
        Returns:
            float: Size impact on delay
        """
        order_size = order.get('size', 0)
        market_size = order.get('market_size', 1)
        
        if market_size == 0:
            return 0
        
        size_ratio = order_size / market_size
        return size_ratio * self.delay_model['size_factor'] * self.delay_model['base_delay']
    
    def _calculate_volatility_impact(self, market_conditions: Dict[str, Any]) -> float:
        """
        Calculate delay impact due to market volatility
        Args:
            market_conditions: Market conditions
        Returns:
            float: Volatility impact on delay
        """
        volatility = market_conditions.get('volatility', 0)
        return volatility * self.delay_model['volatility_factor'] * self.delay_model['base_delay']
    
    def _calculate_liquidity_impact(self, market_conditions: Dict[str, Any]) -> float:
        """
        Calculate delay impact due to market liquidity
        Args:
            market_conditions: Market conditions
        Returns:
            float: Liquidity impact on delay
        """
        liquidity = market_conditions.get('liquidity', 1)
        if liquidity == 0:
            return self.delay_model['base_delay']
        
        return (1 / liquidity) * self.delay_model['liquidity_factor'] * self.delay_model['base_delay']
    
    def _calculate_time_impact(self, order: Dict[str, Any]) -> float:
        """
        Calculate delay impact due to time of day
        Args:
            order: Order details
        Returns:
            float: Time impact multiplier
        """
        timestamp = order.get('timestamp')
        if not timestamp:
            return 1.0
        
        hour = timestamp.hour
        
        # Define market hours (assuming 9:30 AM to 4:00 PM)
        if 9 <= hour < 10 or 15 <= hour < 16:  # Open and close hours
            return self.delay_model['time_of_day_factors']['open']
        else:
            return self.delay_model['time_of_day_factors']['normal']
    
    def _calculate_distribution(self, delays: List[float]) -> Dict[str, float]:
        """
        Calculate delay distribution statistics
        Args:
            delays: List of delay values
        Returns:
            dict: Distribution statistics
        """
        return {
            'percentile_10': np.percentile(delays, 10),
            'percentile_25': np.percentile(delays, 25),
            'percentile_50': np.median(delays),
            'percentile_75': np.percentile(delays, 75),
            'percentile_90': np.percentile(delays, 90)
        }
    
    def _analyze_time_patterns(self, delay_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze time-based patterns in delays
        Args:
            delay_data: List of delay records with timestamps
        Returns:
            dict: Time pattern analysis
        """
        # Group delays by time periods
        hourly_delays = {}
        for record in delay_data:
            hour = record['timestamp'].hour
            if hour not in hourly_delays:
                hourly_delays[hour] = []
            hourly_delays[hour].append(record['delay'])
        
        # Calculate hourly averages
        hourly_avg = {
            hour: np.mean(delays)
            for hour, delays in hourly_delays.items()
        }
        
        return {
            'hourly_averages': hourly_avg,
            'best_hour': min(hourly_avg.items(), key=lambda x: x[1])[0],
            'worst_hour': max(hourly_avg.items(), key=lambda x: x[1])[0]
        }
    
    def _analyze_delay_costs(self, delay_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze costs associated with delays
        Args:
            delay_data: List of delay records
        Returns:
            dict: Cost analysis
        """
        total_costs = sum(d.get('cost_impact', 0) for d in delay_data)
        avg_cost_per_ms = total_costs / sum(d['delay'] for d in delay_data) if delay_data else 0
        
        return {
            'total_costs': total_costs,
            'avg_cost_per_ms': avg_cost_per_ms,
            'max_single_cost': max(d.get('cost_impact', 0) for d in delay_data) if delay_data else 0
        }
