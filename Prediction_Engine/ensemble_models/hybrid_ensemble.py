import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class HybridEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        """
        Initialize the Hybrid Ensemble neural network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output
            dropout: Dropout rate
        """
        super(HybridEnsemble, self).__init__()
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class HybridEnsembleModel:
    def __init__(self, base_models, hidden_dims=[128, 64], output_dim=1, 
                 dropout=0.2, learning_rate=0.001, batch_size=32,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Hybrid Ensemble Model.
        
        Args:
            base_models: Dictionary of model_name: model_object pairs
            hidden_dims: List of hidden layer dimensions for the combining network
            output_dim: Dimension of output
            dropout: Dropout rate
            learning_rate: Learning rate for the combining network
            batch_size: Batch size for training the combining network
            device: Device to use for PyTorch models
        """
        self.base_models = base_models
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        
        # Initialize the combining network
        self.combining_network = None
        self.scaler = StandardScaler()
        
        # Store base model predictions
        self.base_model_predictions = {}
        
    def _get_base_model_predictions(self, X, model):
        """Get predictions from a base model."""
        if hasattr(model, 'predict_proba'):
            if self.output_dim == 1:
                # Binary classification, return probability of positive class
                return model.predict_proba(X)[:, 1].reshape(-1, 1)
            else:
                # Multi-class classification, return all class probabilities
                return model.predict_proba(X)
        else:
            # Regression or models without predict_proba
            return model.predict(X).reshape(-1, 1)
    
    def _generate_base_predictions(self, X):
        """Generate predictions from all base models."""
        predictions = []
        
        for name, model in self.base_models.items():
            pred = self._get_base_model_predictions(X, model)
            predictions.append(pred)
        
        # Concatenate predictions
        return np.hstack(predictions)
    
    def fit(self, X, y, validation_split=0.2, epochs=100):
        """Fit the hybrid ensemble model to the training data."""
        # Generate base model predictions
        base_predictions = self._generate_base_predictions(X)
        
        # Scale the predictions
        base_predictions_scaled = self.scaler.fit_transform(base_predictions)
        
        # Split data for training the combining network
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                base_predictions_scaled, y, test_size=validation_split, random_state=42
            )
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device) if self.output_dim == 1 else torch.LongTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device) if self.output_dim == 1 else torch.LongTensor(y_val).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        else:
            # Convert to tensors
            X_train = torch.FloatTensor(base_predictions_scaled).to(self.device)
            y_train = torch.FloatTensor(y).to(self.device) if self.output_dim == 1 else torch.LongTensor(y).to(self.device)
            
            # Create data loader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            val_loader = None
        
        # Initialize the combining network
        input_dim = base_predictions_scaled.shape[1]
        self.combining_network = HybridEnsemble(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Define loss and optimizer
        criterion = nn.MSELoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.combining_network.parameters(), lr=self.learning_rate)
        
        # Train the combining network
        for epoch in range(epochs):
            self.combining_network.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.combining_network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if val_loader is not None:
                self.combining_network.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.combining_network(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
            else:
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}')
        
        return self
    
    def predict(self, X):
        """Make predictions using the hybrid ensemble model."""
        if self.combining_network is None:
            raise ValueError("Hybrid ensemble model not fitted yet. Call fit() first.")
        
        # Generate base model predictions
        base_predictions = self._generate_base_predictions(X)
        
        # Scale the predictions
        base_predictions_scaled = self.scaler.transform(base_predictions)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(base_predictions_scaled).to(self.device)
        
        # Make predictions
        self.combining_network.eval()
        with torch.no_grad():
            outputs = self.combining_network(X_tensor)
            
            if self.output_dim == 1:
                # Regression or binary classification
                predictions = outputs.cpu().numpy()
                
                # For binary classification, convert probabilities to class labels
                if hasattr(self, 'classes_') and len(self.classes_) == 2:
                    predictions = (predictions > 0.5).astype(int)
            else:
                # Multi-class classification
                _, predicted = torch.max(outputs.data, 1)
                predictions = predicted.cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities for classification tasks."""
        if self.combining_network is None:
            raise ValueError("Hybrid ensemble model not fitted yet. Call fit() first.")
        
        if self.output_dim == 1:
            raise ValueError("predict_proba is only available for multi-class classification tasks.")
        
        # Generate base model predictions
        base_predictions = self._generate_base_predictions(X)
        
        # Scale the predictions
        base_predictions_scaled = self.scaler.transform(base_predictions)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(base_predictions_scaled).to(self.device)
        
        # Make predictions
        self.combining_network.eval()
        with torch.no_grad():
            outputs = self.combining_network(X_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
        
        return probabilities
    
    def save_model(self, filepath):
        """Save the hybrid ensemble model to a file."""
        if self.combining_network is None:
            raise ValueError("Hybrid ensemble model not fitted yet. Call fit() first.")
        
        # Save the combining network state dict
        torch.save(self.combining_network.state_dict(), f"{filepath}_network.pth")
        
        # Save the scaler
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save base models
        for name, model in self.base_models.items():
            if hasattr(model, 'save_model'):
                model.save_model(f"{filepath}_{name}.pkl")
            else:
                joblib.dump(model, f"{filepath}_{name}.pkl")
    
    def load_model(self, filepath):
        """Load a hybrid ensemble model from a file."""
        # Load the combining network state dict
        input_dim = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else len(self.base_models)
        self.combining_network = HybridEnsemble(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        self.combining_network.load_state_dict(torch.load(f"{filepath}_network.pth"))
        
        # Load the scaler
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        # Load base models
        for name in self.base_models.keys():
            if hasattr(self.base_models[name], 'load_model'):
                self.base_models[name].load_model(f"{filepath}_{name}.pkl")
            else:
                self.base_models[name] = joblib.load(f"{filepath}_{name}.pkl")
        
        return self
