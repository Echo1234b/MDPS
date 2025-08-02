import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_predict

class LabelNoiseDetector:
    def __init__(self, features, labels, contamination=0.1):
        self.features = features
        self.labels = labels
        self.contamination = contamination

    def detect_noise(self):
        # Method 1: Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        outlier_predictions = iso_forest.fit_predict(self.features)
        
        # Method 2: Cross-validation prediction consistency
        cv_predictions = cross_val_predict(
            self._get_default_model(),
            self.features,
            self.labels,
            cv=5,
            method='predict_proba'
        )
        
        # Calculate prediction confidence
        confidence = np.max(cv_predictions, axis=1)
        low_confidence_mask = confidence < 0.7
        
        # Combine both methods
        noise_mask = (outlier_predictions == -1) | low_confidence_mask
        
        return noise_mask

    def get_noise_report(self):
        noise_mask = self.detect_noise()
        noise_indices = np.where(noise_mask)[0]
        
        report = {
            'total_samples': len(self.labels),
            'noise_samples': len(noise_indices),
            'noise_percentage': (len(noise_indices) / len(self.labels)) * 100,
            'noise_indices': noise_indices
        }
        
        return report

    def _get_default_model(self):
        # This should be replaced with an actual model implementation
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42)
