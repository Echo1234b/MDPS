import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims=None):
        super(Autoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Encoder layers
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        
        encoder_layers.append(nn.Linear(in_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        in_dim = encoding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims=None):
        super(VariationalAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Encoder layers
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dims[-1], encoding_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], encoding_dim)
        
        # Decoder layers
        decoder_layers = []
        in_dim = encoding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return z, decoded, mu, logvar

class AutoencoderFeatureExtractor:
    def __init__(self, input_dim, encoding_dim=16, hidden_dims=None, 
                 vae=False, learning_rate=0.001,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.vae = vae
        
        if vae:
            self.model = VariationalAutoencoder(
                input_dim=input_dim,
                encoding_dim=encoding_dim,
                hidden_dims=hidden_dims
            ).to(device)
        else:
            self.model = Autoencoder(
                input_dim=input_dim,
                encoding_dim=encoding_dim,
                hidden_dims=hidden_dims
            ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train(self, X_train, epochs=100, batch_size=32, validation_data=None):
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if validation_data is not None:
            X_val = torch.FloatTensor(validation_data).to(self.device)
            val_dataset = TensorDataset(X_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, in train_loader:
                self.optimizer.zero_grad()
                
                if self.vae:
                    encoded, decoded, mu, logvar = self.model(batch_X)
                    
                    # Reconstruction loss
                    recon_loss = F.mse_loss(decoded, batch_X)
                    
                    # KL divergence
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    # Total loss
                    loss = recon_loss + kl_loss / batch_X.size(0)
                else:
                    encoded, decoded = self.model(batch_X)
                    loss = F.mse_loss(decoded, batch_X)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, in val_loader:
                        if self.vae:
                            encoded, decoded, mu, logvar = self.model(batch_X)
                            recon_loss = F.mse_loss(decoded, batch_X)
                            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            loss = recon_loss + kl_loss / batch_X.size(0)
                        else:
                            encoded, decoded = self.model(batch_X)
                            loss = F.mse_loss(decoded, batch_X)
                        
                        val_loss += loss.item()
                
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
            else:
                print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}')
    
    def encode(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            
            if self.vae:
                encoded, _, mu, _ = self.model(X)
                features = mu.cpu().numpy()
            else:
                encoded, _ = self.model(X)
                features = encoded.cpu().numpy()
        
        return features
    
    def decode(self, Z):
        self.model.eval()
        with torch.no_grad():
            Z = torch.FloatTensor(Z).to(self.device)
            decoded = self.model.decoder(Z).cpu().numpy()
        
        return decoded
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
