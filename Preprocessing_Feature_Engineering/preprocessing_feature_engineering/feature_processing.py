import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import List, Dict, Any

class FeatureProcessor:
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
    
    def normalize_features(self, data: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        df = data.copy()
        scaler = self.scalers[method]
        df[columns] = scaler.fit_transform(df[columns])
        return df
    
    def filter_correlated_features(self, data: pd.DataFrame, threshold: float = 0.9) -> List[str]:
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return to_drop
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'k_best', k: int = 10) -> List[str]:
        if method == 'k_best':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()
