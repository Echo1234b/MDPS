import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy import stats

class LabelConsistencyAnalyzer:
    def __init__(self, labels1, labels2=None):
        self.labels1 = labels1
        self.labels2 = labels2

    def analyze_temporal_consistency(self, window_size=30):
        consistency_scores = []
        for i in range(window_size, len(self.labels1)):
            window1 = self.labels1.iloc[i-window_size:i]
            window2 = self.labels1.iloc[i-window_size+1:i+1]
            
            # Calculate agreement between consecutive windows
            agreement = (window1 == window2).mean()
            consistency_scores.append(agreement)
        
        return pd.Series(consistency_scores, index=self.labels1.index[window_size:])

    def compare_label_sets(self):
        if self.labels2 is None:
            raise ValueError("Second label set is required for comparison")
        
        # Calculate Cohen's Kappa
        kappa_score = cohen_kappa_score(self.labels1, self.labels2)
        
        # Calculate agreement rate
        agreement_rate = (self.labels1 == self.labels2).mean()
        
        # Perform statistical test for distribution similarity
        ks_statistic, p_value = stats.ks_2samp(self.labels1, self.labels2)
        
        return {
            'cohen_kappa': kappa_score,
            'agreement_rate': agreement_rate,
            'ks_statistic': ks_statistic,
            'p_value': p_value
        }

    def generate_consistency_report(self):
        temporal_consistency = self.analyze_temporal_consistency()
        comparison_metrics = self.compare_label_sets() if self.labels2 is not None else None
        
        report = {
            'temporal_consistency_mean': temporal_consistency.mean(),
            'temporal_consistency_std': temporal_consistency.std(),
            'comparison_metrics': comparison_metrics
        }
        
        return report
