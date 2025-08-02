import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class CNNExtractor(nn.Module):
    def __init__(self, input_channels, sequence_length, output_size, 
                 kernel_size=3, dropout=0.2):
        super(CNNExtractor, self).__init__()
        
        # Calculate padding to maintain sequence length
        padding = kernel_size // 2
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(64, 128, kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(128, 256, kernel_size, padding=padding)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        # Input shape: (batch_size, input_channels, sequence_length)
        
        # Apply convolutions with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Global max pooling
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNSignalExtractor:
    def __init__(self, input_channels, sequence_length, output_size=1, 
                 kernel_size=3, dropout=0.2, learning_rate=0.001,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = CNNExtractor(
            input_channels=input_channels,
            sequence_length=sequence_length,
            output_size=output_size,
            kernel_size=kernel_size,
            dropout=dropout
        ).to(device)
        self.criterion = nn.MSELoss() if output_size == 1 else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        # Convert to tensors and reshape for CNN
        # X_train shape: (batch_size, sequence_length, input_channels)
        # CNN expects: (batch_size, input_channels, sequence_length)
        X_train = torch.FloatTensor(X_train).permute(0, 2, 1).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = torch.FloatTensor(X_val).permute(0, 2, 1).to(self.device)
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
            # Reshape for CNN
            X = torch.FloatTensor(X).permute(0, 2, 1).to(self.device)
            predictions = self.model(X).cpu().numpy()
        return predictions
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
