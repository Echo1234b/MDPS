from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

class CrossValidationEngine:
    def __init__(self, cv_type='kfold', n_splits=5, random_state=None):
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.random_state = random_state
        
        if cv_type == 'kfold':
            self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        elif cv_type == 'stratified':
            self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        elif cv_type == 'timeseries':
            self.cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV type: {cv_type}")
    
    def evaluate_model(self, model, X, y, task_type='classification'):
        scores = []
        
        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task_type == 'classification':
                score = accuracy_score(y_test, y_pred)
            else:
                score = mean_squared_error(y_test, y_pred)
            
            scores.append(score)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'all_scores': scores
        }
