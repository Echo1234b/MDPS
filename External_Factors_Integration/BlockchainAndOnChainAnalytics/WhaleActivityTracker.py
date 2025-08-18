# WhaleActivityTracker.py
import requests
import pandas as pd

class WhaleActivityTracker:
    def __init__(self, api_url):
        self.api_url = api_url

    def track_whale_activity(self):
        response = requests.get(self.api_url)
        whale_data = response.json()
        return pd.DataFrame(whale_data)

if __name__ == "__main__":
    tracker = WhaleActivityTracker("https://api.example.com/whale_activity")
    whale_data = tracker.track_whale_activity()
    print(whale_data.head())
