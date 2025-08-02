import pandas as pd
import numpy as np

class CandleDirectionLabeler:
    def __init__(self, ohlc_data, body_threshold=0.1):
        self.ohlc_data = ohlc_data
        self.body_threshold = body_threshold

    def generate_direction_labels(self):
        body_size = abs(self.ohlc_data['close'] - self.ohlc_data['open'])
        candle_size = self.ohlc_data['high'] - self.ohlc_data['low']
        
        # Calculate body to candle ratio
        body_ratio = body_size / candle_size
        
        # Generate labels
        labels = np.where(
            (self.ohlc_data['close'] > self.ohlc_data['open']) & (body_ratio > self.body_threshold),
            1,  # Bullish
            np.where(
                (self.ohlc_data['close'] < self.ohlc_data['open']) & (body_ratio > self.body_threshold),
                -1,  # Bearish
                0    # Neutral/Doji
            )
        )
        
        return pd.Series(labels, index=self.ohlc_data.index)

    def get_label_distribution(self):
        labels = self.generate_direction_labels()
        return labels.value_counts(normalize=True)
