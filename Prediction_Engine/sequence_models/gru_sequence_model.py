import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=False):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Get the last time step output
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        return out

class GRUSequenceModel:
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, 
                 learning_rate=0.001, dropout=0.2, bidirectional=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = GRUModel(
            input_size, 
            hidden_size, 
            num_layers, 
            output_size, 
            dropout,
            bidirectional
        ).to(device)
        self.criterion = nn.MSELoss() if output_size == 1 else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
            else:
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}')
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X).cpu().numpy()
        return predictions
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
