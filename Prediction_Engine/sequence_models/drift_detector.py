import numpy as np
from river import drift
import matplotlib.pyplot as plt
from scipy import stats

class ModelDriftDetector:
    def __init__(self, method='adwin', alpha=0.05, window_size=100):
        self.method = method
        self.alpha = alpha
        self.window_size = window_size
        
        # Initialize drift detector based on method
        if method == 'adwin':
            self.detector = drift.ADWIN(delta=alpha)
        elif method == 'ddm':
            self.detector = drift.DDM(min_num_instances=30, warning_threshold=2.0, drift_threshold=3.0)
        elif method == 'eddm':
            self.detector = drift.EDDM(min_num_instances=30, warning_threshold=0.95, drift_threshold=0.9)
        elif method == 'kswin':
            self.detector = drift.KSWIN(alpha=alpha, window_size=window_size, seed=42)
        elif method == 'page_hinkley':
            self.detector = drift.PageHinkley(min_instances=30, delta=0.005, threshold=50.0, alpha=1 - alpha)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
        
        # Track drift history
        self.drift_history = []
        self.warning_history = []
        self.error_history = []
        
    def update(self, error):
        # Update detector
        self.detector.update(error)
        
        # Record error
        self.error_history.append(error)
        
        # Check for drift or warning
        drift_detected = self.detector.drift_detected
        warning_detected = hasattr(self.detector, 'warning_detected') and self.detector.warning_detected
        
        # Record events
        if drift_detected:
            self.drift_history.append(len(self.error
