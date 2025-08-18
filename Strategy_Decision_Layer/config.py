"""
Configuration settings for the Strategy & Decision Layer module.
"""

class Config:
    # Signal Validation Configuration
    SIGNAL_VALIDATION = {
        'min_confidence_threshold': 0.7,
        'max_noise_level': 0.3,
        'redundancy_check': True,
        'validation_window': 100
    }

    # Risk Management Configuration
    RISK_MANAGEMENT = {
        'max_portfolio_risk': 0.02,
        'max_drawdown': 0.15,
        'position_sizing_method': 'fixed_fractional',
        'risk_per_trade': 0.01,
        'max_correlation': 0.7
    }

    # Strategy Selection Configuration
    STRATEGY_SELECTION = {
        'evaluation_window': 30,
        'min_performance_threshold': 0.6,
        'enable_dynamic_selection': True,
        'strategy_weights': {
            'trend_following': 0.3,
            'mean_reversion': 0.3,
            'breakout': 0.4
        }
    }

    # Timing Execution Configuration
    TIMING_EXECUTION = {
        'max_slippage': 0.001,
        'execution_delay_threshold': 100,  # milliseconds
        'enable_microstructure_analysis': True,
        'min_liquidity_threshold': 1000000
    }

    # Simulation Configuration
    SIMULATION = {
        'backtest_period': 365,
        'monte_carlo_runs': 1000,
        'enable_transaction_costs': True,
        'commission_rate': 0.001,
        'slippage_rate': 0.0005
    }
