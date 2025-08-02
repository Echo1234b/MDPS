from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import joblib

class RandomForestPredictor:
    def __init__(self, task_type='classification', n_estimators=100, max_depth=None, random_state=42):
        self.task_type = task_type
        if task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        self.feature_importance = None
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        else:
            mse = mean_squared_error(y_test, y_pred)
            return mse
    
    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
