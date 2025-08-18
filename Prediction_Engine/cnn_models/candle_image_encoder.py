import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import io

class CandleImageEncoder:
    def __init__(self, model_type='resnet', pretrained=True, output_size=1, 
                 dropout=0.2, learning_rate=0.001,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_type = model_type
        
        # Initialize model based on type
        if model_type == 'resnet':
            self.model = models.resnet18(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, output_size)
            )
        elif model_type == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, output_size)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(device)
        self.criterion = nn.MSELoss() if output_size == 1 else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def ohlc_to_image(self, ohlc_data):
        """
        Convert OHLC data to images.
        
        Args:
            ohlc_data: numpy array of shape (sequence_length, 4) with OHLC data
            
        Returns:
            PIL Image
        """
        fig, ax = plt.subplots(figsize=(5, 5), dpi=64)
        
        # Create candlestick chart
        for i, (open_price, high, low, close) in enumerate(ohlc_data):
            color = 'green' if close >= open_price else 'red'
            
            # Draw the vertical line
            ax.plot([i, i], [low, high], color=color, linewidth=1)
            
            # Draw the rectangle
            height = abs(close - open_price)
            bottom = min(open_price, close)
            ax.add_patch(plt.Rectangle((i-0.3, bottom), 0.6, height, color=color))
        
        # Remove axes and whitespace
        ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close()
        
        return img
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        # Convert OHLC data to images
        images = [self.transform(self.ohlc_to_image(sample)) for sample in X_train]
        images = torch.stack(images).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(images, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_images = [self.transform(self.ohlc_to_image(sample)) for sample in X_val]
            val_images = torch.stack(val_images).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(val_images, y_val)
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
            # Convert OHLC data to images
            images = [self.transform(self.ohlc_to_image(sample)) for sample in X]
            images = torch.stack(images).to(self.device)
            
            predictions = self.model(images).cpu().numpy()
        return predictions
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
