# MacroEconomicIndicatorFeed.py
import pandas as pd
import requests

class MacroEconomicIndicatorFeed:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_indicators(self):
        response = requests.get(self.api_url)
        indicators = response.json()
        return pd.DataFrame(indicators)

    def normalize_indicators(self, indicators_df):
        indicators_df['value'] = (indicators_df['value'] - indicators_df['value'].mean()) / indicators_df['value'].std()
        return indicators_df

if __name__ == "__main__":
    feed = MacroEconomicIndicatorFeed("https://api.example.com/macro_indicators")
    indicators = feed.fetch_indicators()
    normalized_indicators = feed.normalize_indicators(indicators)
    print(normalized_indicators.head())
