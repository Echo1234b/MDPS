import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsembleCombiner:
    def __init__(self, models, task_type='classification', method='voting', cv=5):
        """
        Initialize the Ensemble Combiner.
        
        Args:
            models: Dictionary of model_name: model_object pairs
            task_type: 'classification' or 'regression'
            method: 'voting' or 'stacking'
            cv: Number of cross-validation folds
        """
        self.models = models
        self.task_type = task_type
        self.method = method
        self.cv = cv
        self.ensemble = None
        self.scores = None
        
    def fit(self, X, y):
        """Fit the ensemble model to the training data."""
        if self.method == 'voting':
            if self.task_type == 'classification':
                self.ensemble = VotingClassifier(
                    estimators=list(self.models.items()),
                    voting='soft'
                )
            else:
                self.ensemble = VotingRegressor(
                    estimators=list(self.models.items())
                )
        elif self.method == 'stacking':
            if self.task_type == 'classification':
                final_estimator = LogisticRegression()
                self.ensemble = StackingClassifier(
                    estimators=list(self.models.items()),
                    final_estimator=final_estimator,
                    cv=self.cv
                )
            else:
                final_estimator = LinearRegression()
                self.ensemble = StackingRegressor(
                    estimators=list(self.models.items()),
                    final_estimator=final_estimator,
                    cv=self.cv
                )
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        self.ensemble.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble model."""
        if self.ensemble is None:
            raise ValueError("Ensemble model not fitted yet. Call fit() first.")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for classification tasks."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks.")
        if self.ensemble is None:
            raise ValueError("Ensemble model not fitted yet. Call fit() first.")
        return self.ensemble.predict_proba(X)
    
    def evaluate(self, X, y, scoring=None):
        """Evaluate the ensemble model using cross-validation."""
        if self.ensemble is None:
            raise ValueError("Ensemble model not fitted yet. Call fit() first.")
        
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        self.scores = cross_val_score(self.ensemble, X, y, cv=self.cv, scoring=scoring)
        return {
            'mean_score': np.mean(self.scores),
            'std_score': np.std(self.scores),
            'all_scores': self.scores
        }
    
    def get_feature_importance(self):
        """Get feature importance if available."""
        if self.ensemble is None:
            raise ValueError("Ensemble model not fitted yet. Call fit() first.")
        
        if hasattr(self.ensemble, 'feature_importances_'):
            return self.ensemble.feature_importances_
        elif hasattr(self.ensemble, 'coef_'):
            return np.abs(self.ensemble.coef_)
        else:
            raise ValueError("Feature importance not available for this ensemble model.")
    
    def save_model(self, filepath):
        """Save the ensemble model to a file."""
        if self.ensemble is None:
            raise ValueError("Ensemble model not fitted yet. Call fit() first.")
        joblib.dump(self.ensemble, filepath)
    
    def load_model(self, filepath):
        """Load an ensemble model from a file."""
        self.ensemble = joblib.load(filepath)
        return self
