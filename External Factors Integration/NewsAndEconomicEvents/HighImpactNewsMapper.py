# HighImpactNewsMapper.py
import pandas as pd

class HighImpactNewsMapper:
    def __init__(self):
        pass

    def map_news_to_price(self, news_data, price_data):
        merged_data = pd.merge_asof(news_data, price_data, on='timestamp')
        return merged_data

if __name__ == "__main__":
    news_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00']),
        'headline': ['Fed raises rates', 'GDP data released']
    })
    price_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 10:05', '2023-01-01 11:05']),
        'price': [100.0, 101.0]
    })
    mapper = HighImpactNewsMapper()
    mapped_data = mapper.map_news_to_price(news_data, price_data)
    print(mapped_data)
