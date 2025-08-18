from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
import joblib

class SklearnPipeline:
    def __init__(self, model, steps=None):
        if steps is None:
            steps = [
                ('scaler', StandardScaler()),
                ('poly_features', PolynomialFeatures(degree=2)),
                ('feature_selection', SelectKBest(k=10)),
                ('model', model)
            ]
        self.pipeline = Pipeline(steps)
        
    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def grid_search(self, X_train, y_train, param_grid, cv=5):
        grid_search = GridSearchCV(
            self.pipeline, 
            param_grid, 
            cv=cv, 
            n_jobs=-1, 
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.pipeline = grid_search.best_estimator_
        return grid_search.best_params_
    
    def save_pipeline(self, filepath):
        joblib.dump(self.pipeline, filepath)
    
    def load_pipeline(self, filepath):
        self.pipeline = joblib.load(filepath)
