"""
Transaction Cost Modeler implementation.
"""

from typing import Dict, Any, List
import numpy as np
from ..exceptions import SimulationError

class TransactionCostModeler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cost_model = {
            'commission_rate': config['commission_rate'],
            'fixed_costs': 0.0,
            'market_impact': {
                'linear_coefficient': 0.0001,
                'square_root_coefficient': 0.001
            }
        }
    
    def calculate_costs(self, trade_details: Dict[str, Any]) -> float:
        """
        Calculate total transaction costs
        Args:
            trade_details: Trade execution details
        Returns:
            float: Total transaction costs
        """
        try:
            # Commission costs
            commission = self._calculate_commission(trade_details)
            
            # Fixed costs
            fixed_costs = self.cost_model['fixed_costs']
            
            # Market impact costs
            market_impact = self._calculate_market_impact(trade_details)
            
            total_costs = commission + fixed_costs + market_impact
            
            return total_costs
            
        except Exception as e:
            raise SimulationError(f"Cost calculation failed: {str(e)}", trade_details)
    
    def model_impact(self, costs: Dict[str, float], strategy_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Model cost impact on strategy performance
        Args:
            costs: Transaction cost breakdown
            strategy_performance: Strategy performance metrics
        Returns:
            dict: Impact analysis
        """
        try:
            total_costs = sum(costs.values())
            total_return = strategy_performance.get('total_return', 0)
            
            if total_return == 0:
                return {'impact_ratio': 0}
            
            impact_ratio = total_costs / abs(total_return)
            
            return {
                'impact_ratio': impact_ratio,
                'cost_breakdown': costs,
                'performance_after_costs': total_return - total_costs,
                'cost_efficiency': self._calculate_cost_efficiency(costs, strategy_performance)
            }
            
        except Exception as e:
            raise SimulationError(f"Impact modeling failed: {str(e)}", {'costs': costs, 'performance': strategy_performance})
    
    def _calculate_commission(self, trade_details: Dict[str, Any]) -> float:
        """
        Calculate commission costs
        Args:
            trade_details: Trade execution details
        Returns:
            float: Commission costs
        """
        notional_value = trade_details.get('price', 0) * trade_details.get('size', 0)
        return notional_value * self.cost_model['commission_rate']
    
    def _calculate_market_impact(self, trade_details: Dict[str, Any]) -> float:
        """
        Calculate market impact costs
        Args:
            trade_details: Trade execution details
        Returns:
            float: Market impact costs
        """
        trade_size = trade_details.get('size', 0)
        market_volume = trade_details.get('market_volume', 1)
        
        if market_volume == 0:
            return 0
        
        participation_rate = trade_size / market_volume
        
        # Calculate market impact using square root formula
        linear_impact = self.cost_model['market_impact']['linear_coefficient'] * participation_rate
        sqrt_impact = self.cost_model['market_impact']['square_root_coefficient'] * np.sqrt(participation_rate)
        
        return linear_impact + sqrt_impact
    
    def _calculate_cost_efficiency(self, costs: Dict[str, float], performance: Dict[str, float]) -> float:
        """
        Calculate cost efficiency ratio
        Args:
            costs: Transaction cost breakdown
            performance: Strategy performance metrics
        Returns:
            float: Cost efficiency ratio
        """
        total_costs = sum(costs.values())
        total_return = performance.get('total_return', 0)
        
        if total_costs == 0:
            return 1.0
        
        return (total_return - total_costs) / total_return
