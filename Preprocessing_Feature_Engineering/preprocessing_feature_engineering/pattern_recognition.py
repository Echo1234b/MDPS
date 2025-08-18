import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class PatternRecognition:
    def __init__(self):
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'engulfing': self._detect_engulfing
        }
    
    def _detect_doji(self, data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        body = abs(data['close'] - data['open'])
        total_range = data['high'] - data['low']
        return (body / total_range) < threshold
    
    def _detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        body = abs(data['close'] - data['open'])
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        return (lower_shadow > 2 * body) & (upper_shadow < 0.1 * body)
    
    def _detect_engulfing(self, data: pd.DataFrame) -> pd.Series:
        prev_open = data['open'].shift(1)
        prev_close = data['close'].shift(1)
        curr_open = data['open']
        curr_close = data['close']
        
        bullish_engulfing = (prev_close < prev_open) & (curr_close > curr_open) & \
                           (curr_open < prev_close) & (curr_close > prev_open)
        bearish_engulfing = (prev_close > prev_open) & (curr_close < curr_open) & \
                           (curr_open > prev_close) & (curr_close < prev_open)
        
        return bullish_engulfing | bearish_engulfing
    
    def detect_patterns(self, data: pd.DataFrame, patterns: List[str]) -> Dict[str, pd.Series]:
        results = {}
        for pattern in patterns:
            if pattern in self.patterns:
                results[pattern] = self.patterns[pattern](data)
        return results
    
    def encode_patterns(self, patterns: Dict[str, pd.Series]) -> pd.DataFrame:
        return pd.DataFrame(patterns)
