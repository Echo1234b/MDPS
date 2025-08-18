# GeopoliticalRiskIndex.py
import requests
import pandas as pd

class GeopoliticalRiskIndex:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_risk_index(self):
        response = requests.get(self.api_url)
        risk_data = response.json()
        return pd.DataFrame(risk_data)

if __name__ == "__main__":
    index_reader = GeopoliticalRiskIndex("https://api.example.com/geopolitical_risk")
    risk_data = index_reader.fetch_risk_index()
    print(risk_data.head())
