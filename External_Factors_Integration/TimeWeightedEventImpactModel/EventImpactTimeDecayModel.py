# EventImpactTimeDecayModel.py
import pandas as pd
import numpy as np

class EventImpactTimeDecayModel:
    def __init__(self, decay_factor=0.9):
        self.decay_factor = decay_factor

    def apply_decay(self, event_data):
        event_data['decayed_impact'] = event_data['impact'] * (self.decay_factor ** event_data['days_since_event'])
        return event_data

if __name__ == "__main__":
    event_data = pd.DataFrame({
        'event': ['Fed rates', 'GDP data'],
        'impact': [1.0, 0.8],
        'days_since_event': [1, 2]
    })
    decay_model = EventImpactTimeDecayModel()
    decayed_data = decay_model.apply_decay(event_data)
    print(decayed_data)
