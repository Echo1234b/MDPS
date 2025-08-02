import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Calculate attention weights
        attention_weights = self.attention(hidden_states)  # (batch_size, seq_len, 1)
        attention_weights = attention_weights.squeeze(2)   # (batch_size, seq_len)
        
        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states)  # (batch_size, 1, hidden_size)
        context = context.squeeze(1)  # (batch_size, hidden_size)
        
        return context, attention_weights

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, rnn_type='lstm', dropout=0.2):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:  # GRU
            self.rnn = nn.GRU(
                input_size, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        self.attention = AttentionMechanism(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state
        if self.rnn_type.lower() == 'lstm':
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = h0
        
        # Forward propagate RNN
        out, hidden = self.rnn(x, hidden)
        
        # Apply attention
        context, attention_weights = self.attention(out)
        
        # Apply dropout
        context = self.dropout(context)
        
        # Fully connected layer
        out = self.fc(context)
        return out, attention_weights

class AttentionRNNModel:
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, 
                 learning_rate=0.001, dropout=0.2, rnn_type='lstm',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = AttentionRNN(
            input_size, 
            hidden_size, 
            num_layers, 
            output_size, 
            rnn_type,
            dropout
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
                outputs, _ = self.model(batch_X)
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
                        outputs, _ = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
            else:
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}')
    
    def predict(self, X, return_attention=False):
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            predictions, attention_weights = self.model(X)
            predictions = predictions.cpu().numpy()
            attention_weights = attention_weights.cpu().numpy()
        
        if return_attention:
            return predictions, attention_weights
        else:
            return predictions
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
