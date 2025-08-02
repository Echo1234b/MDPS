# GoogleTrendsAPIIntegration.py
from pytrends.request import TrendReq
import pandas as pd

class GoogleTrendsAPIIntegration:
    def __init__(self):
        self.pytrends = TrendReq()

    def fetch_trends(self, keyword, timeframe='today 12-m'):
        self.pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)
        trends_data = self.pytrends.interest_over_time()
        return trends_data

if __name__ == "__main__":
    trends_integration = GoogleTrendsAPIIntegration()
    trends_data = trends_integration.fetch_trends("Bitcoin")
    print(trends_data.head())
