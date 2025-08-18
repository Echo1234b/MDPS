# NewsSentimentAnalyzer.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, news_headlines):
        sentiment_scores = news_headlines.apply(lambda x: self.analyzer.polarity_scores(x))
        return sentiment_scores

if __name__ == "__main__":
    analyzer = NewsSentimentAnalyzer()
    sample_headlines = pd.Series(["Stocks soar after positive earnings report", "Economic downturn feared as inflation rises"])
    sentiment_scores = analyzer.analyze_sentiment(sample_headlines)
    print(sentiment_scores)
