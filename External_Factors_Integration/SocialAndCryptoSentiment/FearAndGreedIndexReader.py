# FearAndGreedIndexReader.py
import requests
import pandas as pd

class FearAndGreedIndexReader:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_index(self):
        response = requests.get(self.api_url)
        index_data = response.json()
        return pd.DataFrame(index_data)

if __name__ == "__main__":
    reader = FearAndGreedIndexReader("https://api.example.com/fear_greed_index")
    index_data = reader.fetch_index()
    print(index_data.head())
