import pandas as pd
import numpy as np

class ThresholdLabeler:
    def __init__(self, data, thresholds=None):
        self.data = data
        self.thresholds = thresholds or {
            'low': 0.01,
            'medium': 0.02,
            'high': 0.03
        }

    def apply_thresholds(self, values):
        labels = []
        for value in values:
            if abs(value) < self.thresholds['low']:
                labels.append('neutral')
            elif abs(value) < self.thresholds['medium']:
                labels.append('low' if value > 0 else 'low_negative')
            elif abs(value) < self.thresholds['high']:
                labels.append('medium' if value > 0 else 'medium_negative')
            else:
                labels.append('high' if value > 0 else 'high_negative')
        
        return pd.Series(labels, index=values.index)

    def create_binary_labels(self, values, threshold=0.0):
        return pd.Series(np.where(values > threshold, 1, 0), index=values.index)

    def create_multi_class_labels(self, values, bins=5):
        return pd.cut(values, bins=bins, labels=[f'class_{i}' for i in range(bins)])
