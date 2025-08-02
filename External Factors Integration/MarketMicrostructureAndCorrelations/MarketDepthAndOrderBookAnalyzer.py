# MarketDepthAndOrderBookAnalyzer.py
import requests
import pandas as pd

class MarketDepthAndOrderBookAnalyzer:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_order_book(self):
        response = requests.get(self.api_url)
        order_book = response.json()
        return pd.DataFrame(order_book)

    def analyze_order_book(self, order_book_df):
        order_book_df['bid_ask_spread'] = order_book_df['ask'] - order_book_df['bid']
        return order_book_df

if __name__ == "__main__":
    analyzer = MarketDepthAndOrderBookAnalyzer("https://api.example.com/order_book")
    order_book = analyzer.fetch_order_book()
    analyzed_order_book = analyzer.analyze_order_book(order_book)
    print(analyzed_order_book.head())
