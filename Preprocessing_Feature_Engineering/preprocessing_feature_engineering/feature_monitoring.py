import pandas as pd
import numpy as np
from typing import Dict, List, Any
import shap
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

class FeatureMonitoring:
    def __init__(self):
        self.feature_importance_history = []
        self.current_version = 0
    
    def log_feature_version(self, features: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        self.current_version += 1
        with mlflow.start_run():
            mlflow.log_param("version", self.current_version)
            mlflow.log_param("feature_count", len(features.columns))
            mlflow.log_params(metadata)
            mlflow.sklearn.log_model(features, f"features_v{self.current_version}")
    
    def track_feature_importance(self, X: pd.DataFrame, y: pd.Series, model: RandomForestClassifier) -> Dict[str, float]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        importance = dict(zip(X.columns, np.abs(shap_values).mean(axis=0)))
        self.feature_importance_history.append(importance)
        
        return importance
    
    def auto_select_features(self, X: pd.DataFrame, y: pd.Series, importance_threshold: float = 0.01) -> List[str]:
        model = RandomForestClassifier()
        model.fit(X, y)
        
        importance = self.track_feature_importance(X, y, model)
        selected_features = [feat for feat, imp in importance.items() if imp > importance_threshold]
        
        return selected_features
