import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class SignalFusionEngine:
    def __init__(self, fusion_method='weighted_average', confidence_threshold=0.5):
        """
        Initialize the Signal Fusion Engine.
        
        Args:
            fusion_method: Method for fusing signals ('weighted_average', 'bayesian', 'voting', 'stacking')
            confidence_threshold: Threshold for signal confidence
        """
        self.fusion_method = fusion_method
        self.confidence_threshold = confidence_threshold
        self.signal_weights = {}
        self.signal_performance = {}
        self.scaler = MinMaxScaler()
        
    def add_signal(self, signal_name, signal_values, signal_confidence=None):
        """
        Add a signal to the fusion engine.
        
        Args:
            signal_name: Name of the signal
            signal_values: Array of signal values
            signal_confidence: Array of confidence scores for the signal (optional)
        """
        if signal_name in self.signal_weights:
            print(f"Warning: Signal {signal_name} already exists. Overwriting.")
        
        # Store signal values
        if not hasattr(self, 'signals'):
            self.signals = {}
        
        self.signals[signal_name] = np.array(signal_values)
        
        # Store signal confidence if provided
        if signal_confidence is not None:
            if not hasattr(self, 'signal_confidences'):
                self.signal_confidences = {}
            self.signal_confidences[signal_name] = np.array(signal_confidence)
        
        # Initialize signal weight if not exists
        if signal_name not in self.signal_weights:
            self.signal_weights[signal_name] = 1.0 / (len(self.signals) if hasattr(self, 'signals') else 1)
    
    def update_signal_weights(self, signal_performance):
        """
        Update signal weights based on performance.
        
        Args:
            signal_performance: Dictionary of signal_name: performance_score pairs
        """
        # Store signal performance
        self.signal_performance = signal_performance
        
        # Normalize performance scores to get weights
        total_performance = sum(signal_performance.values())
        
        if total_performance > 0:
            for signal_name, performance in signal_performance.items():
                self.signal_weights[signal_name] = performance / total_performance
        else:
            # If all performances are zero or negative, use equal weights
            equal_weight = 1.0 / len(signal_performance)
            for signal_name in signal_performance:
                self.signal_weights[signal_name] = equal_weight
    
    def fuse_signals(self, method=None):
        """
        Fuse signals using the specified method.
        
        Args:
            method: Fusion method to use (overrides the default if provided)
            
        Returns:
            Fused signal values
        """
        if not hasattr(self, 'signals') or len(self.signals) == 0:
            raise ValueError("No signals to fuse. Add signals using add_signal() first.")
        
        fusion_method = method if method is not None else self.fusion_method
        
        # Get signal names and values
        signal_names = list(self.signals.keys())
        signal_values = np.array([self.signals[name] for name in signal_names]).T
        
        # Apply fusion method
        if fusion_method == 'weighted_average':
            fused_signal = self._weighted_average_fusion(signal_values, signal_names)
        elif fusion_method == 'bayesian':
            fused_signal = self._bayesian_fusion(signal_values, signal_names)
        elif fusion_method == 'voting':
            fused_signal = self._voting_fusion(signal_values, signal_names)
        elif fusion_method == 'stacking':
            fused_signal = self._stacking_fusion(signal_values, signal_names)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        return fused_signal
    
    def _weighted_average_fusion(self, signal_values, signal_names):
        """Fuse signals using weighted average."""
        # Get weights for each signal
        weights = np.array([self.signal_weights[name] for name in signal_names])
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        fused_signal = np.sum(signal_values * weights, axis=1)
        
        return fused_signal
    
    def _bayesian_fusion(self, signal_values, signal_names):
        """Fuse signals using Bayesian fusion."""
        # Get confidence scores if available
        if hasattr(self, 'signal_confidences'):
            confidences = np.array([self.signal_confidences.get(name, np.ones_like(signal_values[:, 0])) 
                                   for name in signal_names]).T
        else:
            # Use equal confidence if not provided
            confidences = np.ones_like(signal_values)
        
        # Get weights for each signal
        weights = np.array([self.signal_weights[name] for name in signal_names])
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Apply weights to confidences
        weighted_confidences = confidences * weights
        
        # Calculate Bayesian fusion
        fused_signal = np.sum(signal_values * weighted_confidences, axis=1) / np.sum(weighted_confidences, axis=1)
        
        return fused_signal
    
    def _voting_fusion(self, signal_values, signal_names):
        """Fuse signals using voting (for classification signals)."""
        # Get weights for each signal
        weights = np.array([self.signal_weights[name] for name in signal_names])
        
        # Convert signals to binary votes based on threshold
        votes = (signal_values > self.confidence_threshold).astype(int)
        
        # Apply weights to votes
        weighted_votes = votes * weights
        
        # Calculate weighted voting
        fused_signal = np.sum(weighted_votes, axis=1) / np.sum(weights)
        
        return fused_signal
    
    def _stacking_fusion(self, signal_values, signal_names):
        """Fuse signals using a simple stacking approach."""
        # Normalize signal values
        normalized_signals = self.scaler.fit_transform(signal_values)
        
        # Get weights for each signal
        weights = np.array([self.signal_weights[name] for name in signal_names])
        
        # Apply weights to normalized signals
        weighted_signals = normalized_signals * weights
        
        # Calculate stacking fusion
        fused_signal = np.sum(weighted_signals, axis=1)
        
        # Rescale to original range
        fused_signal = self.scaler.inverse_transform(fused_signal.reshape(-1, 1)).flatten()
        
        return fused_signal
    
    def evaluate_fusion(self, true_values, fused_signal, metric='mse'):
        """
        Evaluate the performance of the fused signal.
        
        Args:
            true_values: True values to compare against
            fused_signal: Fused signal values
            metric: Evaluation metric ('mse', 'mae', 'accuracy', 'f1')
            
        Returns:
            Evaluation score
        """
        if metric == 'mse':
            return np.mean((true_values - fused_signal) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(true_values - fused_signal))
        elif metric == 'accuracy':
            binary_true = (true_values > self.confidence_threshold).astype(int)
            binary_fused = (fused_signal > self.confidence_threshold).astype(int)
            return np.mean(binary_true == binary_fused)
        elif metric == 'f1':
            from sklearn.metrics import f1_score
            binary_true = (true_values > self.confidence_threshold).astype(int)
            binary_fused = (fused_signal > self.confidence_threshold).astype(int)
            return f1_score(binary_true, binary_fused)
        else:
            raise ValueError(f"Unknown evaluation metric: {metric}")
    
    def save_fusion_engine(self, filepath):
        """Save the fusion engine to a file."""
        import joblib
        
        # Create a dictionary of all attributes
        fusion_engine_data = {
            'fusion_method': self.fusion_method,
            'confidence_threshold': self.confidence_threshold,
            'signal_weights': self.signal_weights,
            'signal_performance': self.signal_performance,
            'signals': self.signals if hasattr(self, 'signals') else None,
            'signal_confidences': self.signal_confidences if hasattr(self, 'signal_confidences') else None,
            'scaler': self.scaler
        }
        
        # Save to file
        joblib.dump(fusion_engine_data, filepath)
    
    def load_fusion_engine(self, filepath):
        """Load a fusion engine from a file."""
        import joblib
        
        # Load from file
        fusion_engine_data = joblib.load(filepath)
        
        # Set attributes
        self.fusion_method = fusion_engine_data['fusion_method']
        self.confidence_threshold = fusion_engine_data['confidence_threshold']
        self.signal_weights = fusion_engine_data['signal_weights']
        self.signal_performance = fusion_engine_data['signal_performance']
        
        if fusion_engine_data['signals'] is not None:
            self.signals = fusion_engine_data['signals']
        
        if fusion_engine_data['signal_confidences'] is not None:
            self.signal_confidences = fusion_engine_data['signal_confidences']
        
        self.scaler = fusion_engine_data['scaler']
        
        return self
