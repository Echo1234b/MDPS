import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelSelector:
    def __init__(self, models, task_type='classification', cv=5, scoring=None):
        """
        Initialize the Model Selector.
        
        Args:
            models: Dictionary of model_name: model_object pairs
            task_type: 'classification' or 'regression'
            cv: Number of cross-validation folds
            scoring: Scoring metric for model evaluation
        """
        self.models = models
        self.task_type = task_type
        self.cv = cv
        self.scoring = scoring
        
        # Default scoring based on task type
        if scoring is None:
            self.scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
        
        # Store model performance
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf if task_type == 'classification' else np.inf
        
        # Store model predictions
        self.model_predictions = {}
        
        # Store model metadata
        self.model_metadata = {}
    
    def evaluate_models(self, X, y):
        """
        Evaluate all models using cross-validation.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary of model_name: evaluation_score pairs
        """
        self.model_scores = {}
        
        for name, model in self.models.items():
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
                
                # Store mean and std of scores
                self.model_scores[name] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'all_scores': scores
                }
                
                # Update best model if needed
                if self.task_type == 'classification':
                    if self.model_scores[name]['mean_score'] > self.best_score:
                        self.best_score = self.model_scores[name]['mean_score']
                        self.best_model = model
                        self.best_model_name = name
                else:
                    if self.model_scores[name]['mean_score'] < self.best_score:
                        self.best_score = self.model_scores[name]['mean_score']
                        self.best_model = model
                        self.best_model_name = name
                
            except Exception as e:
                print(f"Error evaluating model {name}: {str(e)}")
                self.model_scores[name] = {
                    'mean_score': -np.inf if self.task_type == 'classification' else np.inf,
                    'std_score': np.nan,
                    'all_scores': []
                }
        
        return self.model_scores
    
    def fit_best_model(self, X, y):
        """
        Fit the best model to the training data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Fitted model
        """
        if self.best_model is None:
            raise ValueError("No best model found. Call evaluate_models() first.")
        
        self.best_model.fit(X, y)
        return self.best_model
    
    def predict(self, X, model_name=None):
        """
        Make predictions using a specific model or the best model.
        
        Args:
            X: Feature data
            model_name: Name of the model to use (if None, use the best model)
            
        Returns:
            Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model found. Call evaluate_models() first.")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found.")
            model = self.models[model_name]
        
        predictions = model.predict(X)
        
        # Store predictions if needed
        if model_name is None:
            model_name = self.best_model_name
        
        self.model_predictions[model_name] = predictions
        
        return predictions
    
    def predict_proba(self, X, model_name=None):
        """
        Predict class probabilities for classification tasks.
        
        Args:
            X: Feature data
            model_name: Name of the model to use (if None, use the best model)
            
        Returns:
            Class probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks.")
        
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model found. Call evaluate_models() first.")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found.")
            model = self.models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {model_name} does not support predict_proba.")
        
        return model.predict_proba(X)
    
    def evaluate_on_test(self, X_test, y_test, model_name=None, metrics=None):
        """
        Evaluate a model on test data.
        
        Args:
            X_test: Test feature data
            y_test: Test target data
            model_name: Name of the model to evaluate (if None, use the best model)
            metrics: List of metrics to compute (if None, use default metrics)
            
        Returns:
            Dictionary of metric_name: score pairs
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model found. Call evaluate_models() first.")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found.")
            model = self.models[model_name]
        
        # Make predictions
        predictions = self.predict(X_test, model_name)
        
        # Default metrics based on task type
        if metrics is None:
            if self.task_type == 'classification':
                metrics = ['accuracy', 'f1', 'precision', 'recall']
            else:
                metrics = ['mse', 'mae', 'rmse']
        
        # Compute metrics
        scores = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                scores[metric] = accuracy_score(y_test, predictions)
            elif metric == 'f1':
                scores[metric] = f1_score(y_test, predictions, average='weighted')
            elif metric == 'precision':
                scores[metric] = precision_score(y_test, predictions, average='weighted')
            elif metric == 'recall':
                scores[metric] = recall_score(y_test, predictions, average='weighted')
            elif metric == 'mse':
                scores[metric] = mean_squared_error(y_test, predictions)
            elif metric == 'mae':
                scores[metric] = np.mean(np.abs(y_test - predictions))
            elif metric == 'rmse':
                scores[metric] = np.sqrt(mean_squared_error(y_test, predictions))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        # Store scores
        if model_name not in self.model_metadata:
            self.model_metadata[model_name] = {}
        
        self.model_metadata[model_name]['test_scores'] = scores
        
        return scores
    
    def get_model_ranking(self):
        """
        Get a ranking of models based on their performance.
        
        Returns:
            DataFrame with model ranking
        """
        if not self.model_scores:
            raise ValueError("No model scores found. Call evaluate_models() first.")
        
        # Create DataFrame with model scores
        ranking_data = []
        for name, scores in self.model_scores.items():
            ranking_data.append({
                'model_name': name,
                'mean_score': scores['mean_score'],
                'std_score': scores['std_score']
            })
        
        # Sort by mean score
        if self.task_type == 'classification':
            ranking_data.sort(key=lambda x: x['mean_score'], reverse=True)
        else:
            ranking_data.sort(key=lambda x: x['mean_score'])
        
        # Create DataFrame
        ranking_df = pd.DataFrame(ranking_data)
        
        # Add rank column
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df
    
    def save_model_selector(self, filepath):
        """Save the model selector to a file."""
        # Create a dictionary of all attributes
        model_selector_data = {
            'models': self.models,
            'task_type': self.task_type,
            'cv': self.cv,
            'scoring': self.scoring,
            'model_scores': self.model_scores,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'model_predictions': self.model_predictions,
            'model_metadata': self.model_metadata
        }
        
        # Save to file
        joblib.dump(model_selector_data, filepath)
    
    def load_model_selector(self, filepath):
        """Load a model selector from a file."""
        # Load from file
        model_selector_data = joblib.load(filepath)
        
        # Set attributes
        self.models = model_selector_data['models']
        self.task_type = model_selector_data['task_type']
        self.cv = model_selector_data['cv']
        self.scoring = model_selector_data['scoring']
        self.model_scores = model_selector_data['model_scores']
        self.best_model_name = model_selector_data['best_model_name']
        self.best_score = model_selector_data['best_score']
        self.model_predictions = model_selector_data['model_predictions']
        self.model_metadata = model_selector_data['model_metadata']
        
        # Set best model
        if self.best_model_name is not None:
            self.best_model = self.models[self.best_model_name]
        
        return self
