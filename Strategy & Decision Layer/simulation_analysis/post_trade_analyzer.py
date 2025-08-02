"""
Post Trade Analyzer implementation.
"""

from typing import Dict, Any, List
import numpy as np
from datetime import datetime
from ..exceptions import SimulationError

class PostTradeAnalyzer:
    def __init__(self):
        self.analysis_metrics = {
            'performance': ['return', 'sharpe_ratio', 'max_drawdown'],
            'execution': ['slippage', 'timing', 'costs'],
            'risk': ['var', 'exposure', 'correlation']
        }
    
    def analyze_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze executed trade
        Args:
            trade_data: Trade execution data
        Returns:
            dict: Analysis results
        """
        try:
            analysis = {
                'performance': self._analyze_performance(trade_data),
                'execution': self._analyze_execution(trade_data),
                'risk': self._analyze_risk(trade_data),
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            raise SimulationError(f"Trade analysis failed: {str(e)}", trade_data)
    
    def _analyze_performance(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze trade performance metrics
        Args:
            trade_data: Trade execution data
        Returns:
            dict: Performance metrics
        """
        entry_price = trade_data.get('entry_price', 0)
        exit_price = trade_data.get('exit_price', 0)
        position_size = trade_data.get('position_size', 0)
        costs = trade_data.get('costs', 0)
        
        gross_pnl = (exit_price - entry_price) * position_size
        net_pnl = gross_pnl - costs
        
        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return': net_pnl / (entry_price * position_size) if entry_price * position_size != 0 else 0,
            'cost_ratio': costs / abs(gross_pnl) if gross_pnl != 0 else 0
        }
    
    def _analyze_execution(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze trade execution quality
        Args:
            trade_data: Trade execution data
        Returns:
            dict: Execution metrics
        """
        intended_price = trade_data.get('intended_price', 0)
        executed_price = trade_data.get('executed_price', 0)
        execution_delay = trade_data.get('execution_delay', 0)
        
        slippage = abs(executed_price - intended_price) / intended_price if intended_price != 0 else 0
        
        return {
            'slippage': slippage,
            'execution_delay': execution_delay,
            'execution_score': 1 - min(slippage + execution_delay / 1000, 1)  # Normalize to 0-1
        }
    
    def _analyze_risk(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze trade risk metrics
        Args:
            trade_data: Trade execution data
        Returns:
            dict: Risk metrics
        """
        entry_price = trade_data.get('entry_price', 0)
        position_size = trade_data.get('position_size', 0)
        stop_loss = trade_data.get('stop_loss', 0)
        
        risk_amount = abs(entry_price - stop_loss) * position_size
        portfolio_value = trade_data.get('portfolio_value', 1)
        
        return {
            'trade_risk': risk_amount,
            'portfolio_risk': risk_amount / portfolio_value if portfolio_value != 0 else 0,
            'risk_reward_ratio': trade_data.get('take_profit', 0) / risk_amount if risk_amount != 0 else 0
        }
    
    def generate_report(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate analysis report from multiple trade analyses
        Args:
            analysis_results: List of trade analysis results
        Returns:
            dict: Summary report
        """
        try:
            if not analysis_results:
                return {}
            
            # Aggregate metrics
            aggregated_metrics = {
                'performance': self._aggregate_performance_metrics(analysis_results),
                'execution': self._aggregate_execution_metrics(analysis_results),
                'risk': self._aggregate_risk_metrics(analysis_results)
            }
            
            # Calculate summary statistics
            summary = {
                'total_trades': len(analysis_results),
                'winning_trades': len([r for r in analysis_results if r['performance']['net_pnl'] > 0]),
                'total_pnl': sum(r['performance']['net_pnl'] for r in analysis_results),
                'average_return': np.mean([r['performance']['return'] for r in analysis_results]),
                'sharpe_ratio': self._calculate_sharpe_ratio(analysis_results),
                'max_drawdown': self._calculate_max_drawdown(analysis_results)
            }
            
            return {
                'summary': summary,
                'aggregated_metrics': aggregated_metrics,
                'recommendations': self._generate_recommendations(aggregated_metrics)
            }
            
        except Exception as e:
            raise SimulationError(f"Report generation failed: {str(e)}", {'results_count': len(analysis_results)})
    
    def _aggregate_performance_metrics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate performance metrics from multiple analyses"""
        returns = [r['performance']['return'] for r in analysis_results]
        pnls = [r['performance']['net_pnl'] for r in analysis_results]
        
        return {
            'avg_return': np.mean(returns),
            'return_std': np.std(returns),
            'total_pnl': sum(pnls),
            'win_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0
        }
    
    def _aggregate_execution_metrics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate execution metrics from multiple analyses"""
        slippages = [r['execution']['slippage'] for r in analysis_results]
        delays = [r['execution']['execution_delay'] for r in analysis_results]
        
        return {
            'avg_slippage': np.mean(slippages),
            'avg_delay': np.mean(delays),
            'execution_score': np.mean([r['execution']['execution_score'] for r in analysis_results])
        }
    
    def _aggregate_risk_metrics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate risk metrics from multiple analyses"""
        trade_risks = [r['risk']['trade_risk'] for r in analysis_results]
        portfolio_risks = [r['risk']['portfolio_risk'] for r in analysis_results]
        
        return {
            'avg_trade_risk': np.mean(trade_risks),
            'avg_portfolio_risk': np.mean(portfolio_risks),
            'max_portfolio_risk': max(portfolio_risks) if portfolio_risks else 0
        }
    
    def _calculate_sharpe_ratio(self, analysis_results: List[Dict[str, Any]]) -> float:
        """Calculate Sharpe ratio from analysis results"""
        returns = [r['performance']['return'] for r in analysis_results]
        if not returns or np.std(returns) == 0:
            return 0
        
        return np.sqrt(252) * np.mean(returns) / np.std(returns)
    
    def _calculate_max_drawdown(self, analysis_results: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown from analysis results"""
        cumulative_returns = np.cumsum([r['performance']['return'] for r in analysis_results])
        if not len(cumulative_returns):
            return 0
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        
        return max(drawdown) if len(drawdown) > 0 else 0
    
    def _generate_recommendations(self, aggregated_metrics: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        # Performance recommendations
        if aggregated_metrics['performance']['win_rate'] < 0.5:
            recommendations.append("Consider reviewing entry/exit criteria to improve win rate")
        
        if aggregated_metrics['performance']['return_std'] > 0.2:
            recommendations.append("High return volatility detected, consider position sizing adjustments")
        
        # Execution recommendations
        if aggregated_metrics['execution']['avg_slippage'] > 0.001:
            recommendations.append("High slippage detected, consider optimizing execution timing")
        
        if aggregated_metrics['execution']['avg_delay'] > 100:
            recommendations.append("Execution delays detected, consider broker or infrastructure improvements")
        
