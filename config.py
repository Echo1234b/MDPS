"""
MDPS Configuration Module
Central configuration management for all MDPS components.
"""

from pathlib import Path
import json

class MDPSConfig:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        
        # Data Collection Settings
        self.mt5_settings = {
            "server": "MetaQuotes-Demo",
            "timeout": 60000,
            "reconnect_attempts": 3
        }
        
        # Data Cleaning Settings
        self.cleaning_settings = {
            "remove_outliers": True,
            "fill_missing": "forward",
            "smooth_window": 5
        }
        
        # Feature Engineering Settings
        self.feature_settings = {
            "technical_indicators": [
                "SMA", "EMA", "RSI", "MACD", "BB"
            ],
            "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"]
        }
        
        # Market Analysis Settings
        self.market_analysis = {
            "support_resistance_periods": 20,
            "volatility_window": 14,
            "trend_threshold": 0.6
        }
        
        # Strategy Settings
        self.strategy_settings = {
            "risk_per_trade": 0.02,
            "max_open_positions": 3,
            "stop_loss_atr_factor": 2.0
        }
    
    def save_config(self, filepath):
        """Save current configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load_config(cls, filepath):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        config = cls()
        config.__dict__.update(config_data)
        return config
