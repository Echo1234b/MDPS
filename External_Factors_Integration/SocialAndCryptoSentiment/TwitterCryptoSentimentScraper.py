# TwitterCryptoSentimentScraper.py
import tweepy
import pandas as pd

class TwitterCryptoSentimentScraper:
    def __init__(self, api_key, api_secret_key, access_token, access_token_secret):
        self.auth = tweepy.OAuthHandler(api_key, api_secret_key)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)

    def scrape_tweets(self, query, count=100):
        tweets = tweepy.Cursor(self.api.search_tweets, q=query, lang="en").items(count)
        tweet_data = [[tweet.created_at, tweet.text] for tweet in tweets]
        return pd.DataFrame(tweet_data, columns=['timestamp', 'text'])

if __name__ == "__main__":
    scraper = TwitterCryptoSentimentScraper("API_KEY", "API_SECRET", "ACCESS_TOKEN", "ACCESS_SECRET")
    tweets = scraper.scrape_tweets("Bitcoin", count=10)
    print(tweets.head())
