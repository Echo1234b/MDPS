import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
import optuna
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    def __init__(self, model, param_space, task_type='classification', cv=5, 
                 scoring=None, n_trials=50, search_type='optuna', random_state=42):
        """
        Initialize the Hyperparameter Tuner.
        
        Args:
            model: Model to tune
            param_space: Parameter search space
            task_type: 'classification' or 'regression'
            cv: Number of cross-validation folds
            scoring: Scoring metric for model evaluation
            n_trials: Number of trials for random search or Optuna
            search_type: Type of search ('grid', 'random', 'optuna')
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.param_space = param_space
        self.task_type = task_type
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.search_type = search_type
        self.random_state = random_state
        
        # Default scoring based on task type
        if scoring is None:
            self.scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
        
        # Store search results
        self.search_results = None
        self.best_params = None
        self.best_score = -np.inf if task_type == 'classification' else np.inf
        self.best_model = None
        
        # Optuna study
        self.study = None
    
    def tune(self, X, y):
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Best parameters and best score
        """
        if self.search_type == 'grid':
            self._grid_search(X, y)
        elif self.search_type == 'random':
            self._random_search(X, y)
        elif self.search_type == 'optuna':
            self._optuna_search(X, y)
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
        
        return self.best_params, self.best_score
    
    def _grid_search(self, X, y):
        """Perform grid search."""
        # Create grid search object
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_space,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Store results
        self.search_results = grid_search.cv_results_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_model = grid_search.best_estimator_
    
    def _random_search(self, X, y):
        """Perform random search."""
        # Create random search object
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_space,
            n_iter=self.n_trials,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit random search
        random_search.fit(X, y)
        
        # Store results
        self.search_results = random_search.cv_results_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_model = random_search.best_estimator_
    
    def _optuna_search(self, X, y):
        """Perform Optuna search."""
        # Define objective function
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in self.param_space.items():
                if isinstance(param_values, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Numerical parameter
                    if isinstance(param_values[0], int):
                        # Integer parameter
                        params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
                    else:
                        # Float parameter
                        params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
                else:
                    raise ValueError(f"Invalid parameter values for {param_name}: {param_values}")
            
            # Handle special cases for different model types
            if isinstance(self.model, (xgb.XGBClassifier, xgb.XGBRegressor)):
                # XGBoost model
                model = type(self.model)(**params, random_state=self.random_state)
            elif isinstance(self.model, type) and issubclass(self.model, nn.Module):
                # PyTorch model
                return self._evaluate_pytorch_model(trial, X, y, params)
            else:
                # Scikit-learn model
                model = type(self.model)(**params, random_state=self.random_state)
            
            # Evaluate model
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            return np.mean(scores)
        
        # Create study
        direction = 'maximize' if self.task_type == 'classification' else 'minimize'
        self.study = optuna.create_study(direction=direction)
        
        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials)
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # Create best model
        if isinstance(self.model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            self.best_model = type(self.model)(**self.best_params, random_state=self.random_state)
            self.best_model.fit(X, y)
        elif isinstance(self.model, type) and issubclass(self.model, nn.Module):
            self.best_model = self._create_pytorch_model(self.best_params)
            self._train_pytorch_model(self.best_model, X, y, self.best_params)
        else:
            self.best_model = type(self.model)(**self.best_params, random_state=self.random_state)
            self.best_model.fit(X, y)
    
    def _create_pytorch_model(self, params):
        """Create a PyTorch model with the given parameters."""
        # This should be implemented based on the specific PyTorch model architecture
        # For now, we'll return a simple MLP as an example
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim, dropout):
                super(SimpleMLP, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, output_dim))
                
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        # Get input and output dimensions
        input_dim = params.get('input_dim', 10)
        output_dim = params.get('output_dim', 1)
        
        # Get hidden dimensions
        hidden_dims = params.get('hidden_dims', [64, 32])
        
        # Get dropout
        dropout = params.get('dropout', 0.2)
        
        # Create model
        model = SimpleMLP(input_dim, hidden_dims, output_dim, dropout)
        
        return model
    
    def _evaluate_pytorch_model(self, trial, X, y, params):
        """Evaluate a PyTorch model with the given parameters."""
        # Create model
        model = self._create_pytorch_model(params)
        
        # Get training parameters
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.001)
        epochs = params.get('epochs', 10)
        
        # Train model
        score = self._train_pytorch_model(model, X, y, params, trial=trial)
        
        return score
    
    def _train_pytorch_model(self, model, X, y, params, trial=None):
        """Train a PyTorch model and return the evaluation score."""
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Get training parameters
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.001)
        epochs = params.get('epochs', 10)
        
        # Convert data to tensors
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device) if self.task_type == 'regression' else torch.LongTensor(y).to(device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss and optimizer
        criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Report intermediate result if using Optuna
            if trial is not None:
                trial.report(loss.item(), epoch)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            
            if self.task_type == 'regression':
                score = -F.mse_loss(outputs, y_tensor).item()  # Negative MSE for maximization
            else:
                _, predicted = torch.max(outputs.data, 1)
                score = accuracy_score(y_tensor.cpu().numpy(), predicted.cpu().numpy())
        
        return score
    
    def get_search_results(self):
        """
        Get the search results.
        
        Returns:
            DataFrame with search results
        """
        if self.search_results is None:
            raise ValueError("No search results found. Call tune() first.")
        
        if self.search_type == 'optuna':
            # Convert Optuna study results to DataFrame
            trials_df = self.study.trials_dataframe()
            return trials_df
        else:
            # Convert sklearn search results to DataFrame
            results_df = pd.DataFrame(self.search_results)
            
            # Select relevant columns
            param_columns = [col for col in results_df.columns if col.startswith('param_')]
            selected_columns = param_columns + ['mean_test_score', 'std_test_score', 'rank_test_score']
            
            return results_df[selected_columns].sort_values('rank_test_score')
    
    def get_best_model(self):
        """
        Get the best model found during the search.
        
        Returns:
            Best model
        """
        if self.best_model is None:
            raise ValueError("No best model found. Call tune() first.")
        
        return self.best_model
    
    def save_results(self, filepath):
        """Save the search results to a file."""
        if self.search_results is None:
            raise ValueError("No search results found. Call tune() first.")
        
        # Create a dictionary of all attributes
        tuner_data = {
            'model_type': type(self.model).__name__,
            'param_space': self.param_space,
            'task_type': self.task_type,
            'cv': self.cv,
            'scoring': self.scoring,
            'n_trials': self.n_trials,
            'search_type': self.search_type,
            'random_state': self.random_state,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'search_results': self.get_search_results().to_dict() if self.search_type != 'optuna' else None
        }
        
        # Save Optuna study if available
        if self.study is not None:
            tuner_data['optuna_study'] = self.study
        
        # Save to file
        joblib.dump(tuner_data, filepath)
        
        # Save best model separately
        if self.best_model is not None:
            if isinstance(self.best_model, nn.Module):
                torch.save(self.best_model.state_dict(), f"{filepath}_model.pth")
            else:
                joblib.dump(self.best_model, f"{filepath}_model.pkl")
    
    def load_results(self, filepath):
        """Load search results from a file."""
        # Load from file
        tuner_data = joblib.load(filepath)
        
        # Set attributes
        self.param_space = tuner_data['param_space']
        self.task_type = tuner_data['task_type']
        self.cv = tuner_data['cv']
        self.scoring = tuner_data['scoring']
        self.n_trials = tuner_data['n_trials']
        self.search_type = tuner_data['search_type']
        self.random_state = tuner_data['random_state']
        self.best_params = tuner_data['best_params']
        self.best_score = tuner_data['best_score']
        
        # Load Optuna study if available
        if 'optuna_study' in tuner_data:
            self.study = tuner_data['optuna_study']
        
        # Load search results if available
        if tuner_data['search_results'] is not None:
            self.search_results = tuner_data['search_results']
        
        # Load best model if available
        try:
            if isinstance(self.model, nn.Module):
                self.best_model = self._create_pytorch_model(self.best_params)
                self.best_model.load_state_dict(torch.load(f"{filepath}_model.pth"))
            else:
                self.best_model = joblib.load(f"{filepath}_model.pkl")
        except:
            print("Warning: Could not load best model.")
        
        return self
