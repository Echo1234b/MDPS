# FundingRateMonitor.py
import requests
import pandas as pd

class FundingRateMonitor:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_funding_rates(self):
        response = requests.get(self.api_url)
        funding_rates = response.json()
        return pd.DataFrame(funding_rates)

    def analyze_rates(self, funding_rates_df):
        funding_rates_df['timestamp'] = pd.to_datetime(funding_rates_df['timestamp'])
        return funding_rates_df

if __name__ == "__main__":
    monitor = FundingRateMonitor("https://api.example.com/funding_rates")
    funding_rates = monitor.fetch_funding_rates()
    analyzed_rates = monitor.analyze_rates(funding_rates)
    print(analyzed_rates.head())
