"""
Trade Simulator implementation.
"""

from typing import Dict, Any, List
import numpy as np
from ..exceptions import SimulationError

class TradeSimulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.simulation_params = {
            'enable_costs': config['enable_transaction_costs'],
            'commission_rate': config['commission_rate'],
            'slippage_rate': config['slippage_rate']
        }
        self.slippage_simulator = SlippageSimulator(config)
        self.cost_modeler = TransactionCostModeler(config)
        self.delay_emulator = ExecutionDelayEmulator(config)
    
    def simulate_trade(self, strategy: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate trade execution
        Args:
            strategy: Trading strategy parameters
            market_data: Market data for simulation
        Returns:
            dict: Simulation results
        """
        try:
            # Initialize simulation
            simulation = {
                'entry': self._simulate_entry(strategy, market_data),
                'exit': None,
                'costs': {},
                'performance': {}
            }
            
            # Simulate trade lifecycle
            if simulation['entry']['executed']:
                simulation['exit'] = self._simulate_exit(strategy, market_data, simulation['entry'])
                simulation['costs'] = self._calculate_total_costs(simulation)
                simulation['performance'] = self._calculate_performance(simulation)
            
            return simulation
            
        except Exception as e:
            raise SimulationError(f"Trade simulation failed: {str(e)}", strategy)
    
    def _simulate_entry(self, strategy: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate trade entry
        Args:
            strategy: Trading strategy parameters
            market_data: Market data
        Returns:
            dict: Entry simulation results
        """
        entry_price = strategy.get('entry_price')
        entry_size = strategy.get('position_size')
        
        # Simulate execution delay
        delay = self.delay_emulator.emulate_delay(strategy, market_data)
        
        # Simulate slippage
        slippage = self.slippage_simulator.simulate_slippage(
            {'price': entry_price, 'size': entry_size},
            market_data
        )
        
        executed_price = entry_price + slippage
        
        return {
            'price': executed_price,
            'size': entry_size,
            'timestamp': market_data['timestamp'] + delay,
            'executed': True,
            'slippage': slippage,
            'delay': delay
        }
    
    def _simulate_exit(self, strategy: Dict[str, Any], market_data: Dict[str, Any], 
                      entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate trade exit
        Args:
            strategy: Trading strategy parameters
            market_data: Market data
            entry: Entry simulation results
        Returns:
            dict: Exit simulation results
        """
        exit_price = self._calculate_exit_price(strategy, market_data, entry)
        exit_size = entry['size']
        
        # Simulate execution delay
        delay = self.delay_emulator.emulate_delay(strategy, market_data)
        
        # Simulate slippage
        slippage = self.slippage_simulator.simulate_slippage(
            {'price': exit_price, 'size': exit_size},
            market_data
        )
        
        executed_price = exit_price + slippage
        
        return {
            'price': executed_price,
            'size': exit_size,
            'timestamp': market_data['timestamp'] + delay,
            'executed': True,
            'slippage': slippage,
            'delay': delay
        }
    
    def _calculate_exit_price(self, strategy: Dict[str, Any], market_data: Dict[str, Any], 
                            entry: Dict[str, Any]) -> float:
        """
        Calculate exit price based on strategy rules
        Args:
            strategy: Trading strategy parameters
            market_data: Market data
            entry: Entry simulation results
        Returns:
            float: Exit price
        """
        direction = strategy.get('direction')
        stop_loss = strategy.get('stop_loss')
        take_profit = strategy.get('take_profit')
        current_price = market_data.get('current_price')
        
        if direction == 'long':
            if current_price <= stop_loss:
                return stop_loss
            elif current_price >= take_profit:
                return take_profit
        else:
            if current_price >= stop_loss:
                return stop_loss
            elif current_price <= take_profit:
                return take_profit
        
        return current_price
    
    def _calculate_total_costs(self, simulation: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate total transaction costs
        Args:
            simulation: Trade simulation data
        Returns:
            dict: Cost breakdown
        """
        if not self.simulation_params['enable_costs']:
            return {'total': 0}
        
        entry_costs = self.cost_modeler.calculate_costs({
            'price': simulation['entry']['price'],
            'size': simulation['entry']['size']
        })
        
        exit_costs = self.cost_modeler.calculate_costs({
            'price': simulation['exit']['price'],
            'size': simulation['exit']['size']
        })
        
        return {
            'entry_costs': entry_costs,
            'exit_costs': exit_costs,
            'slippage_costs': (
                abs(simulation['entry']['slippage']) +
                abs(simulation['exit']['slippage'])
            ) * simulation['entry']['size'],
            'total': entry_costs + exit_costs + (
                abs(simulation['entry']['slippage']) +
                abs(simulation['exit']['slippage'])
            ) * simulation['entry']['size']
        }
    
    def _calculate_performance(self, simulation: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate trade performance metrics
        Args:
            simulation: Trade simulation data
        Returns:
            dict: Performance metrics
        """
        entry_price = simulation['entry']['price']
        exit_price = simulation['exit']['price']
        size = simulation['entry']['size']
        total_costs = simulation['costs']['total']
        
        gross_pnl = (exit_price - entry_price) * size
        net_pnl = gross_pnl - total_costs
        
        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return': net_pnl / (entry_price * size),
            'costs_ratio': total_costs / abs(gross_pnl) if gross_pnl != 0 else 0
        }
