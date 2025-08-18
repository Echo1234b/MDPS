# CorrelatedAssetTracker.py
import pandas as pd
import numpy as np

class CorrelatedAssetTracker:
    def __init__(self):
        pass

    def track_correlation(self, asset_data):
        correlation_matrix = asset_data.corr()
        return correlation_matrix

if __name__ == "__main__":
    asset_data = pd.DataFrame({
        'asset1': [1, 2, 3, 4, 5],
        'asset2': [5, 4, 3, 2, 1]
    })
    tracker = CorrelatedAssetTracker()
    correlation_matrix = tracker.track_correlation(asset_data)
    print(correlation_matrix)
