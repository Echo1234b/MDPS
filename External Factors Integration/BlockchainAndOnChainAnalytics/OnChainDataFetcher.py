# OnChainDataFetcher.py
import requests
import pandas as pd

class OnChainDataFetcher:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_on_chain_data(self):
        response = requests.get(self.api_url)
        on_chain_data = response.json()
        return pd.DataFrame(on_chain_data)

if __name__ == "__main__":
    fetcher = OnChainDataFetcher("https://api.example.com/on_chain_data")
    on_chain_data = fetcher.fetch_on_chain_data()
    print(on_chain_data.head())
