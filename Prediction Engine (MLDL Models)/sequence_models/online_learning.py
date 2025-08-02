import numpy as np
from river import compose, linear_model, optim, preprocessing, metrics
from river import drift
import joblib
import warnings
warnings.filterwarnings('ignore')

class OnlineLearningUpdater:
    def __init__(self, model_type='linear_regression', learning_rate=0.01):
        self.model_type = model_type
        self.learning_rate = learning_rate
        
        # Initialize model based on type
        if model_type == 'linear_regression':
            self.model = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LinearRegression(
                    optimizer=optim.SGD(learning_rate)
                )
            )
        elif model_type == 'logistic_regression':
            self.model = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression(
                    optimizer=optim.SGD(learning_rate)
                )
            )
        elif model_type == 'hoeffding_tree':
            self.model = compose.Pipeline(
                preprocessing.StandardScaler(),
                tree.HoeffdingTreeClassifier()
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize metrics
        self.metrics = {
            'mae': metrics.MAE(),
            'mse': metrics.MSE(),
            'rmse': metrics.RMSE(),
            'r2': metrics.R2()
        } if model_type == 'linear_regression' else {
            'accuracy': metrics.Accuracy(),
            'f1': metrics.F1(),
            'precision': metrics.Precision(),
            'recall': metrics.Recall()
        }
        
        # Initialize drift detector
        self.drift_detector = drift.ADWIN()
        self.drift_detected = False
        
        # Track performance
        self.performance_history = []
        
    def update(self, x, y):
        # Make prediction
        y_pred = self.model.predict_one(x)
        
        # Update metrics
        for metric in self.metrics.values():
            metric.update(y, y_pred)
        
        # Update model
        self.model.learn_one(x, y)
        
        # Check for drift
        if self.model_type == 'linear_regression':
            error = abs(y - y_pred)
        else:
            error = 0 if y == y_pred else 1
        
        self.drift_detected = self.drift_detector.update(error)
        
        # Record performance
        current_metrics = {name: metric.get() for name, metric in self.metrics.items()}
        self.performance_history.append(current_metrics)
        
        return y_pred, self.drift_detected
    
    def get_metrics(self):
        return {name: metric.get() for name, metric in self.metrics.items()}
    
    def get_performance_history(self):
        return self.performance_history
    
    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.revert(self.metrics['mae'].n)
    
    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
