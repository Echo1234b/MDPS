import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
    
    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q: [B, H, L, D]
        # K: [B, H, S, D]
        B, H, L, D = Q.shape
        _, _, S, _ = K.shape
        
        # Calculate the sampled Q and K
        K_expand = K.unsqueeze(-3).expand(B, H, L, S, D)
        index_sample = torch.randint(low=0, high=S, size=(L, sample_k))
        K_sample = K_expand[:, :, torch.arange(L).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        
        # Find the Top_k query with the largest similarity
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        
        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        
        # Calculate attention scores
        attn_scores, idx = self._prob_QK(queries, keys, sample_k=U, n_top=self.factor)
        
        # Apply mask if needed
        if self.mask_flag:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, H, 1, 1)
            attn_scores.masked_fill_(attn_mask, -np.inf)
        
        # Softmax
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        V = torch.zeros_like(queries)
        for i in range(B):
            for j in range(H):
                V[i, :, j, :] = torch.matmul(attn[i, j, :, :], values[i, :, j, :])
        
        return V.transpose(2, 1), idx

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.attention = attention
        self.query_projection = nn.Linear(d_model, d_model * n_heads)
        self.key_projection = nn.Linear(d_model, d_model * n_heads)
        self.value_projection = nn.Linear(d_model, d_model * n_heads)
        self.out_projection = nn.Linear(d_model * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Linear projections
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # Apply attention
        out, attn = self.attention(
            queries, keys, values, attn_mask
        )
        
        # Output projection
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        
        return out, attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None):
        # Self-attention
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        
        # Feed-forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns

class Informer(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, e_layers, d_ff, dropout, attn='prob', factor=5):
        super(Informer, self).__init__()
        
        # Encoder
        self.encoder_embedding = nn.Linear(enc_in, d_model)
        
        # Attention
        Attn = ProbAttention(mask_flag=False, factor=factor, dropout=dropout)
        
        # Encoder Layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn, d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu"
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Projection
        self.projection = nn.Linear(d_model, 1, bias=True)
        
    def forward(self, x_enc, x_mark_enc=None, enc_self_mask=None):
        # Embedding
        enc_out = self.encoder_embedding(x_enc)
        
        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # Projection
        dec_out = self.projection(enc_out)
        
        return dec_out[:, -1:, :], attns

class InformerTransformer:
    def __init__(self, enc_in, d_model=512, n_heads=8, e_layers=3, d_ff=2048, 
                 dropout=0.1, factor=5, learning_rate=0.0001,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = Informer(
            enc_in=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn='prob',
            factor=factor
        ).to(device)
        self.criterion = nn.MSELoss()
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
            attention_weights = [attn.cpu().numpy() for attn in attention_weights]
        
        if return_attention:
            return predictions, attention_weights
        else:
            return predictions
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
