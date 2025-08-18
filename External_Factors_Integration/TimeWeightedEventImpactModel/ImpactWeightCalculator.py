# ImpactWeightCalculator.py
import pandas as pd
import numpy as np

class ImpactWeightCalculator:
    def __init__(self):
        pass

    def calculate_weights(self, event_data):
        event_data['weight'] = event_data['impact'] / event_data['impact'].sum()
        return event_data

if __name__ == "__main__":
    event_data = pd.DataFrame({
        'event': ['Fed rates', 'GDP data'],
        'impact': [1.0, 0.8]
    })
    weight_calculator = ImpactWeightCalculator()
    weighted_data = weight_calculator.calculate_weights(event_data)
    print(weighted_data)
