# SocialMediaSentimentTracker.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SocialMediaSentimentTracker:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def track_sentiment(self, posts):
        sentiment_scores = posts.apply(lambda x: self.analyzer.polarity_scores(x))
        return sentiment_scores

if __name__ == "__main__":
    tracker = SocialMediaSentimentTracker()
    sample_posts = pd.Series(["Bitcoin is going to the moon!", "Crypto market is crashing"])
    sentiment_scores = tracker.track_sentiment(sample_posts)
    print(sentiment_scores)
