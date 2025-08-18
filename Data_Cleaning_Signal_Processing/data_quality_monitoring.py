# data_processing/data_quality_monitoring.py
"""
Data Quality Monitoring & Drift Detection Module

This module provides tools for detecting concept drift, monitoring distribution changes,
and analyzing data quality.
"""

import pandas as pd
import numpy as np
from scipy import stats
from river import drift
from river import compose
from river import preprocessing
from river import metrics

class ConceptDriftDetector:
    def __init__(self):
        self.drift_detector = drift.DDM()
        self.metric = metrics.Accuracy()
        
    def update(self, y_true, y_pred):
        self.drift_detector.update(y_true, y_pred)
        return self.drift_detector.drift_detected

class DistributionChangeMonitor:
    @staticmethod
    def kolmogorov_smirnov_test(data1, data2):
        return stats.ks_2samp(data1, data2)
    
    @staticmethod
    def jensen_shannon_divergence(p, q):
        p = np.array(p)
        q = np.array(q)
        p = p / np.sum(p)
        q = q / np.sum(q)
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p/m)) + np.sum(q * np.log(q/m)))

class DataQualityAnalyzer:
    @staticmethod
    def analyze_missing_values(df):
        return df.isnull().sum()
    
    @staticmethod
    def analyze_duplicates(df):
        return df.duplicated().sum()
    
    @staticmethod
    def analyze_outliers(df, threshold=3):
        z_scores = stats.zscore(df.select_dtypes(include=[np.number]))
        abs_z_scores = np.abs(z_scores)
        return (abs_z_scores > threshold).sum()
