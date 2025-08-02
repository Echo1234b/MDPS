# SentimentAggregator.py
import pandas as pd

class SentimentAggregator:
    def __init__(self):
        pass

    def aggregate_sentiment(self, sentiment_data):
        aggregated = sentiment_data.groupby('timestamp')['sentiment_score'].mean()
        return aggregated

if __name__ == "__main__":
    sentiment_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:00', '2023-01-01 11:00']),
        'sentiment_score': [0.5, -0.2, 0.8]
    })
    aggregator = SentimentAggregator()
    aggregated_sentiment = aggregator.aggregate_sentiment(sentiment_data)
    print(aggregated_sentiment)
