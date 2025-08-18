# data_processing/contextual_structural_annotation.py
"""
Contextual and Structural Annotation Module

This module provides tools for annotating price actions, classifying market phases,
mapping events, enriching context, and detecting behavioral anomalies.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

class PriceActionAnnotator:
    @staticmethod
    def annotate_breakouts(df, window=20):
        df['breakout'] = (df['close'] > df['close'].rolling(window=window).max())
        return df
    
    @staticmethod
    def annotate_pullbacks(df, window=20):
        df['pullback'] = (df['close'] < df['close'].rolling(window=window).min())
        return df

class MarketPhaseClassifier:
    @staticmethod
    def classify_market_phases(df, n_clusters=3):
        features = df[['close', 'volume', 'high', 'low']].pct_change()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['market_phase'] = kmeans.fit_predict(features.dropna())
        return df

class EventMapper:
    @staticmethod
    def map_events(df, events_df):
        return pd.merge_asof(df, events_df, left_index=True, right_on='event_time', direction='forward')

class ContextEnricher:
    @staticmethod
    def enrich_context(df):
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['trend'] = df['close'].rolling(window=20).mean()
        df['volatility_ratio'] = df['volatility'] / df['trend']
        return df

class BehavioralPatternAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination)
    
    def detect_anomalies(self, df):
        features = df[['close', 'volume', 'high', 'low']].pct_change()
        df['anomaly'] = self.model.fit_predict(features.dropna())
        return df
