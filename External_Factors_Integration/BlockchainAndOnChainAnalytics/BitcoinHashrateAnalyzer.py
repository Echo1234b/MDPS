# BitcoinHashrateAnalyzer.py
import requests
import pandas as pd

class BitcoinHashrateAnalyzer:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_hashrate(self):
        response = requests.get(self.api_url)
        hashrate_data = response.json()
        return pd.DataFrame(hashrate_data)

    def analyze_hashrate(self, hashrate_df):
        hashrate_df['timestamp'] = pd.to_datetime(hashrate_df['timestamp'])
        return hashrate_df

if __name__ == "__main__":
    analyzer = BitcoinHashrateAnalyzer("https://api.example.com/bitcoin_hashrate")
    hashrate_data = analyzer.fetch_hashrate()
    analyzed_hashrate = analyzer.analyze_hashrate(hashrate_data)
    print(analyzed_hashrate.head())
