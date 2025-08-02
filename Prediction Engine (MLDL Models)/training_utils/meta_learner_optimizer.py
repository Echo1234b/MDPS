import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import optuna
import warnings
warnings.filterwarnings('ignore')

class MetaLearnerOptimizer:
    def __init__(self, models, task_type='classification', cv=5, scoring=None, 
                 n_trials=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Meta-Learner Optimizer.
        
        Args:
            models: Dictionary of model_name: model_object pairs
            task_type: 'classification' or 'regression'
            cv: Number of cross-validation folds
            scoring: Scoring metric for model evaluation
            n_trials: Number of trials for hyperparameter optimization
            device: Device to use for PyTorch models
        """
        self.models = models
        self.task_type = task_type
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.device = device
        
        # Default scoring based on task type
        if scoring is None:
            self.scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
        
        # Store model performance
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf if task_type == 'classification' else np.inf
        
        # Store hyperparameter search spaces
        self.param_spaces = self._define_param_spaces()
        
        # Store optimized hyperparameters
        self.best_params = {}
        
        # Store meta-features
        self.meta_features = None
        self.meta_targets = None
        
        # Store meta-model
        self.meta_model = None
    
    def _define_param_spaces(self):
        """Define hyperparameter search spaces for each model type."""
        param_spaces = {}
        
        # XGBoost
        if 'xgboost' in self.models:
            param_spaces['xgboost'] = {
                'n_estimators': (50, 500),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'gamma': (0, 5),
                'reg_alpha': (0, 5),
                'reg_lambda': (0, 5)
            }
        
        # Random Forest
        if 'random_forest' in self.models:
            param_spaces['random_forest'] = {
                'n_estimators': (50, 500),
                'max_depth': (3, 30),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': (0.1, 1.0)
            }
        
        # Neural Network
        if 'neural_network' in self.models:
            param_spaces['neural_network'] = {
                'hidden_layer_sizes': [(64,), (128,), (256,), (64, 64), (128, 64), (128, 128)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': (0.0001, 0.1),
                'learning_rate_init': (0.001, 0.1)
            }
        
        # LSTM
        if 'lstm' in self.models:
            param_spaces['lstm'] = {
                'hidden_size': (32, 256),
                'num_layers': (1, 3),
                'dropout': (0.1, 0.5),
                'learning_rate': (0.0001, 0.01)
            }
        
        # Transformer
        if 'transformer' in self.models:
            param_spaces['transformer'] = {
                'd_model': (128, 512),
                'nhead': (2, 8),
                'num_encoder_layers': (1, 4),
                'dim_feedforward': (256, 2048),
                'dropout': (0.1, 0.5),
                'learning_rate': (0.0001, 0.01)
            }
        
        return param_spaces
    
    def optimize_all_models(self, X, y):
        """
        Optimize hyperparameters for all models.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary of model_name: (best_params, best_score) pairs
        """
        results = {}
        
        for model_name, model in self.models.items():
            if model_name in self.param_spaces:
                print(f"Optimizing {model_name}...")
                
                if model_name == 'xgboost':
                    best_params, best_score = self._optimize_xgboost(X, y)
                elif model_name == 'random_forest':
                    best_params, best_score = self._optimize_random_forest(X, y)
                elif model_name == 'neural_network':
                    best_params, best_score = self._optimize_neural_network(X, y)
                elif model_name == 'lstm':
                    best_params, best_score = self._optimize_lstm(X, y)
                elif model_name == 'transformer':
                    best_params, best_score = self._optimize_transformer(X, y)
                else:
                    print(f"Warning: No optimization method defined for {model_name}")
                    continue
                
                results[model_name] = (best_params, best_score)
                self.best_params[model_name] = best_params
                
                # Update best model if needed
                if self.task_type == 'classification':
                    if best_score > self.best_score:
                        self.best_score = best_score
                        self.best_model_name = model_name
                else:
                    if best_score < self.best_score:
                        self.best_score = best_score
                        self.best_model_name = model_name
        
        # Set best model with optimized parameters
        if self.best_model_name is not None:
            self.best_model = clone(self.models[self.best_model_name])
            self.best_model.set_params(**self.best_params[self.best_model_name])
            self.best_model.fit(X, y)
        
        return results
    
    def _optimize_xgboost(self, X, y):
        """Optimize XGBoost hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *self.param_spaces['xgboost']['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *self.param_spaces['xgboost']['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *self.param_spaces['xgboost']['learning_rate']),
                'subsample': trial.suggest_float('subsample', *self.param_spaces['xgboost']['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *self.param_spaces['xgboost']['colsample_bytree']),
                'gamma': trial.suggest_float('gamma', *self.param_spaces['xgboost']['gamma']),
                'reg_alpha': trial.suggest_float('reg_alpha', *self.param_spaces['xgboost']['reg_alpha']),
                'reg_lambda': trial.suggest_float('reg_lambda', *self.param_spaces['xgboost']['reg_lambda'])
            }
            
            if self.task_type == 'classification':
                model = xgb.XGBClassifier(**params, random_state=42)
            else:
                model = xgb.XGBRegressor(**params, random_state=42)
            
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize' if self.task_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params, study.best_value
    
    def _optimize_random_forest(self, X, y):
        """Optimize Random Forest hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *self.param_spaces['random_forest']['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *self.param_spaces['random_forest']['max_depth']),
                'min_samples_split': trial.suggest_int('min_samples_split', *self.param_spaces['random_forest']['min_samples_split']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *self.param_spaces['random_forest']['min_samples_leaf']),
                'max_features': trial.suggest_float('max_features', *self.param_spaces['random_forest']['max_features'])
            }
            
            model = clone(self.models['random_forest'])
            model.set_params(**params)
            
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize' if self.task_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params, study.best_value
    
    def _optimize_neural_network(self, X, y):
        """Optimize Neural Network hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', self.param_spaces['neural_network']['hidden_layer_sizes']),
                'activation': trial.suggest_categorical('activation', self.param_spaces['neural_network']['activation']),
                'alpha': trial.suggest_float('alpha', *self.param_spaces['neural_network']['alpha']),
                'learning_rate_init': trial.suggest_float('learning_rate_init', *self.param_spaces['neural_network']['learning_rate_init'])
            }
            
            model = clone(self.models['neural_network'])
            model.set_params(**params)
            
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize' if self.task_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params, study.best_value
    
    def _optimize_lstm(self, X, y):
        """Optimize LSTM hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'hidden_size': trial.suggest_int('hidden_size', *self.param_spaces['lstm']['hidden_size']),
                'num_layers': trial.suggest_int('num_layers', *self.param_spaces['lstm']['num_layers']),
                'dropout': trial.suggest_float('dropout', *self.param_spaces['lstm']['dropout']),
                'learning_rate': trial.suggest_float('learning_rate', *self.param_spaces['lstm']['learning_rate'])
            }
            
            # Create a simple LSTM model for optimization
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                    super(SimpleLSTM, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    out, _ = self.lstm(x, (h0, c0))
                    out = out[:, -1, :]
                    out = self.fc(out)
                    return out
            
            # Convert data to PyTorch tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device) if self.task_type == 'regression' else torch.LongTensor(y).to(self.device)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Initialize model
            input_size = X.shape[2]  # Assuming X is 3D: (batch_size, seq_len, features)
            output_size = 1 if self.task_type == 'regression' else len(np.unique(y))
            model = SimpleLSTM(input_size, params['hidden_size'], params['num_layers'], output_size, params['dropout']).to(self.device)
            
            # Define loss and optimizer
            criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Training
            model.train()
            for epoch in range(10):  # Short training for optimization
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
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
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params, study.best_value
    
    def _optimize_transformer(self, X, y):
        """Optimize Transformer hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'd_model': trial.suggest_int('d_model', *self.param_spaces['transformer']['d_model']),
                'nhead': trial.suggest_int('nhead', *self.param_spaces['transformer']['nhead']),
                'num_encoder_layers': trial.suggest_int('num_encoder_layers', *self.param_spaces['transformer']['num_encoder_layers']),
                'dim_feedforward': trial.suggest_int('dim_feedforward', *self.param_spaces['transformer']['dim_feedforward']),
                'dropout': trial.suggest_float('dropout', *self.param_spaces['transformer']['dropout']),
                'learning_rate': trial.suggest_float('learning_rate', *self.param_spaces['transformer']['learning_rate'])
            }
            
            # Create a simple Transformer model for optimization
            class SimpleTransformer(nn.Module):
                def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_size):
                    super(SimpleTransformer, self).__init__()
                    self.input_projection = nn.Linear(input_size, d_model)
                    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
                    self.output_projection = nn.Linear(d_model, output_size)
                
                def forward(self, x):
                    # x shape: (batch_size, seq_len, input_size)
                    x = self.input_projection(x)
                    x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
                    x = self.transformer_encoder(x)
                    x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
                    x = x[:, -1, :]  # Get the last output
                    x = self.output_projection(x)
                    return x
            
            # Convert data to PyTorch tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device) if self.task_type == 'regression' else torch.LongTensor(y).to(self.device)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Initialize model
            input_size = X.shape[2]  # Assuming X is 3D: (batch_size, seq_len, features)
            output_size = 1 if self.task_type == 'regression' else len(np.unique(y))
            model = SimpleTransformer(
                input_size, 
                params['d_model'], 
                params['nhead'], 
                params['num_encoder_layers'], 
                params['dim_feedforward'], 
                params['dropout'], 
                output_size
            ).to(self.device)
            
            # Define loss and optimizer
            criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Training
            model.train()
            for epoch in range(10):  # Short training for optimization
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
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
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params, study.best_value
    
    def extract_meta_features(self, X, y):
        """
        Extract meta-features from the dataset.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Meta-features
        """
        # Simple statistical meta-features
        meta_features = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)) if self.task_type == 'classification' else 0,
            'ratio_features_to_samples': X.shape[1] / X.shape[0],
            'mean_feature_value': np.mean(X),
            'std_feature_value': np.std(X),
            'skewness_feature_value': pd.DataFrame(X).skew().mean(),
            'kurtosis_feature_value': pd.DataFrame(X).kurtosis().mean(),
            'mean_correlation': pd.DataFrame(X).corr().mean().mean(),
            'std_correlation': pd.DataFrame(X).corr().std().std()
        }
        
        # Additional meta-features for classification
        if self.task_type == 'classification':
            class_counts = np.bincount(y)
            meta_features['class_imbalance'] = np.max(class_counts) / np.sum(class_counts)
            meta_features['entropy'] = -np.sum((class_counts / np.sum(class_counts)) * np.log2(class_counts / np.sum(class_counts)))
        
        self.meta_features = meta_features
        return meta_features
    
    def train_meta_model(self, meta_features_list, meta_targets):
        """
        Train a meta-model to select the best base model for a given dataset.
        
        Args:
            meta_features_list: List of meta-features for multiple datasets
            meta_targets: List of best model names for each dataset
            
        Returns:
            Trained meta-model
        """
        # Convert meta-features to DataFrame
        meta_df = pd.DataFrame(meta_features_list)
        
        # Create a mapping from model names to integers
        model_names = list(set(meta_targets))
        model_name_to_id = {name: i for i, name in enumerate(model_names)}
        
        # Convert targets to integers
        meta_targets_int = [model_name_to_id[name] for name in meta_targets]
        
        # Train a simple meta-model
        if self.task_type == 'classification':
            self.meta_model = xgb.XGBClassifier(random_state=42)
        else:
            self.meta_model = xgb.XGBRegressor(random_state=42)
        
        self.meta_model.fit(meta_df, meta_targets_int)
        
        return self.meta_model
    
    def select_best_model(self, X, y):
        """
        Select the best model for the given dataset using the meta-model.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Best model name
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_meta_model() first.")
        
        # Extract meta-features
        meta_features = self.extract_meta_features(X, y)
        
        # Convert to DataFrame
        meta_df = pd.DataFrame([meta_features])
        
        # Predict best model
        model_id = self.meta_model.predict(meta_df)[0]
        
        # Get model name
        model_names = list(self.models.keys())
        best_model_name = model_names[model_id]
        
        return best_model_name
    
    def save_meta_learner(self, filepath):
        """Save the meta-learner to a file."""
        # Create a dictionary of all attributes
        meta_learner_data = {
            'models': self.models,
            'task_type': self.task_type,
            'cv': self.cv,
            'scoring': self.scoring,
            'n_trials': self.n_trials,
            'param_spaces': self.param_spaces,
            'best_params': self.best_params,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'meta_features': self.meta_features,
            'meta_model': self.meta_model
        }
        
        # Save to file
        joblib.dump(meta_learner_data, filepath)
    
    def load_meta_learner(self, filepath):
        """Load a meta-learner from a file."""
        # Load from file
        meta_learner_data = joblib.load(filepath)
        
        # Set attributes
        self.models = meta_learner_data['models']
        self.task_type = meta_learner_data['task_type']
        self.cv = meta_learner_data['cv']
        self.scoring = meta_learner_data['scoring']
        self.n_trials = meta_learner_data['n_trials']
        self.param_spaces = meta_learner_data['param_spaces']
        self.best_params = meta_learner_data['best_params']
        self.best_model_name = meta_learner_data['best_model_name']
        self.best_score = meta_learner_data['best_score']
        self.meta_features = meta_learner_data['meta_features']
        self.meta_model = meta_learner_data['meta_model']
        
        # Set best model with optimized parameters
        if self.best_model_name is not None:
            self.best_model = clone(self.models[self.best_model_name])
            self.best_model.set_params(**self.best_params[self.best_model_name])
        
        return self
