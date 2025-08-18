# EventImpactEstimator.py
import pandas as pd
import numpy as np

class EventImpactEstimator:
    def __init__(self):
        pass

    def estimate_impact(self, event_data, price_data):
        event_data['price_change'] = price_data['price'].pct_change()
        impact = event_data.groupby('event_type')['price_change'].mean()
        return impact

if __name__ == "__main__":
    event_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00']),
        'event_type': ['Fed rates', 'GDP data']
    })
    price_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00']),
        'price': [100.0, 101.0]
    })
    estimator = EventImpactEstimator()
    impact = estimator.estimate_impact(event_data, price_data)
    print(impact)
