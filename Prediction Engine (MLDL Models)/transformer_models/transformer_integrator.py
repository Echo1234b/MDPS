import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, 
                 dim_feedforward=2048, dropout=0.1, output_dim=1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.d_model = d_model
    
    def forward(self, src, src_mask=None):
        # src shape: (seq_len, batch_size, input_dim)
        
        # Project input to d_model dimensions
        src = self.input_projection(src) * np.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Project to output dimension
        output = self.output_projection(output)
        
        # Return the last time step output
        return output[-1, :, :]

class TransformerIntegrator:
    def __init__(self, input_dim, d_model=512, nhead=8, num_encoder_layers=3, 
                 dim_feedforward=2048, dropout=0.1, output_dim=1, learning_rate=0.0001,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = TransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim
        ).to(device)
        self.criterion = nn.MSELoss() if output_dim == 1 else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        # Convert to tensors and reshape for Transformer
        # X_train shape: (batch_size, seq_len, input_dim)
        # Transformer expects: (seq_len, batch_size, input_dim)
        X_train = torch.FloatTensor(X_train).permute(1, 0, 2).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = torch.FloatTensor(X_val).permute(1, 0, 2).to(self.device)
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
            # Reshape for Transformer
            X = torch.FloatTensor(X).permute(1, 0, 2).to(self.device)
            predictions = self.model(X).cpu().numpy()
        return predictions
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
