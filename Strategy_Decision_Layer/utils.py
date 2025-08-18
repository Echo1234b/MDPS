"""
Utility functions for the Strategy & Decision Layer.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate various performance metrics
    Args:
        returns: Series of returns
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {
        'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),
        'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min(),
        'win_rate': len(returns[returns > 0]) / len(returns),
        'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
        'total_return': returns.sum()
    }
    return metrics

def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """
    Convert timestamp to datetime object
    Args:
        timestamp: Unix timestamp
    Returns:
        datetime object
    """
    return datetime.fromtimestamp(timestamp)

def normalize_data(data: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Normalize data using z-score
    Args:
        data: Input data to normalize
    Returns:
        Normalized data
    """
    return (data - data.mean()) / data.std()

def calculate_correlation(returns1: pd.Series, returns2: pd.Series) -> float:
    """
    Calculate correlation between two return series
    Args:
        returns1: First return series
        returns2: Second return series
    Returns:
        Correlation coefficient
    """
    return returns1.corr(returns2)

def rolling_metrics(returns: pd.Series, window: int = 30) -> pd.DataFrame:
    """
    Calculate rolling performance metrics
    Args:
        returns: Series of returns
        window: Rolling window size
    Returns:
        DataFrame of rolling metrics
    """
    metrics = pd.DataFrame(index=returns.index)
    metrics['rolling_sharpe'] = returns.rolling(window).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std()
    )
    metrics['rolling_drawdown'] = returns.rolling(window).apply(
        lambda x: (x.cumsum() - x.cumsum().cummax()).min()
    )
    metrics['rolling_win_rate'] = returns.rolling(window).apply(
        lambda x: len(x[x > 0]) / len(x)
    )
    return metrics
