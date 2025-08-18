#!/usr/bin/env python3
"""
MDPS System Configuration
Centralized configuration with validation, environment management, and feature flags
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "mdps_db"
    username: str = "mdps_user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    timeout: int = 30
    max_requests: int = 1000

@dataclass
class TradingConfig:
    """Trading system configuration"""
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    default_stop_loss: float = 0.02
    default_take_profit: float = 0.04
    max_drawdown: float = 0.15
    trading_enabled: bool = False
    paper_trading: bool = True

@dataclass
class DataConfig:
    """Data collection and processing configuration"""
    data_dir: str = "data"
    cache_dir: str = "cache"
    backup_dir: str = "backup"
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB
    data_retention_days: int = 365
    compression_enabled: bool = True
    real_time_updates: bool = True

@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_dir: str = "models"
    training_data_dir: str = "training_data"
    validation_split: float = 0.2
    test_split: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10

@dataclass
class UIConfig:
    """User interface configuration"""
    theme: str = "dark"
    language: str = "en"
    auto_save: bool = True
    auto_refresh: bool = True
    refresh_interval: int = 1000  # milliseconds
    max_chart_data_points: int = 10000

@dataclass
class SystemConfig:
    """System-wide configuration"""
    log_level: str = "INFO"
    log_file: str = "mdps.log"
    max_log_size: int = 1024 * 1024 * 100  # 100MB
    log_backup_count: int = 5
    performance_monitoring: bool = True
    health_check_interval: int = 30  # seconds
    backup_interval: int = 3600  # 1 hour

class MDPSConfig:
    """Main configuration manager for MDPS system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.yaml"
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize default configurations
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.ml = MLConfig()
        self.ui = UIConfig()
        self.system = SystemConfig()
        
        # Load configuration
        self.load_config()
        
    def load_config(self):
        """Load configuration from file or environment variables"""
        try:
            # Try to load from YAML file
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self._update_from_dict(config_data)
                    logger.info(f"Configuration loaded from {self.config_file}")
            else:
                # Load from environment variables
                self._load_from_env()
                logger.info("Configuration loaded from environment variables")
                
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}. Using defaults.")
            
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, data in config_data.items():
            if hasattr(self, section) and isinstance(data, dict):
                section_obj = getattr(self, section)
                for key, value in data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Database
        if os.getenv("MDPS_DB_HOST"):
            self.database.host = os.getenv("MDPS_DB_HOST")
        if os.getenv("MDPS_DB_PORT"):
            self.database.port = int(os.getenv("MDPS_DB_PORT"))
        if os.getenv("MDPS_DB_NAME"):
            self.database.database = os.getenv("MDPS_DB_NAME")
        if os.getenv("MDPS_DB_USER"):
            self.database.username = os.getenv("MDPS_DB_USER")
        if os.getenv("MDPS_DB_PASS"):
            self.database.password = os.getenv("MDPS_DB_PASS")
            
        # API
        if os.getenv("MDPS_API_HOST"):
            self.api.host = os.getenv("MDPS_API_HOST")
        if os.getenv("MDPS_API_PORT"):
            self.api.port = int(os.getenv("MDPS_API_PORT"))
        if os.getenv("MDPS_API_DEBUG"):
            self.api.debug = os.getenv("MDPS_API_DEBUG").lower() == "true"
            
        # Trading
        if os.getenv("MDPS_TRADING_ENABLED"):
            self.trading.trading_enabled = os.getenv("MDPS_TRADING_ENABLED").lower() == "true"
        if os.getenv("MDPS_PAPER_TRADING"):
            self.trading.paper_trading = os.getenv("MDPS_PAPER_TRADING").lower() == "true"
            
        # System
        if os.getenv("MDPS_LOG_LEVEL"):
            self.system.log_level = os.getenv("MDPS_LOG_LEVEL")
            
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'database': asdict(self.database),
                'api': asdict(self.api),
                'trading': asdict(self.trading),
                'data': asdict(self.data),
                'ml': asdict(self.ml),
                'ui': asdict(self.ui),
                'system': asdict(self.system)
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            
    def get_config(self, section: str) -> Any:
        """Get configuration section"""
        if hasattr(self, section):
            return getattr(self, section)
        raise ValueError(f"Configuration section '{section}' not found")
        
    def update_config(self, section: str, key: str, value: Any):
        """Update specific configuration value"""
        if hasattr(self, section):
            section_obj = getattr(self, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                logger.info(f"Updated {section}.{key} = {value}")
            else:
                raise ValueError(f"Configuration key '{key}' not found in section '{section}'")
        else:
            raise ValueError(f"Configuration section '{section}' not found")
            
    def validate_config(self) -> bool:
        """Validate current configuration"""
        try:
            # Validate database connection
            if not self.database.host or not self.database.database:
                logger.error("Invalid database configuration")
                return False
                
            # Validate API settings
            if self.api.port < 1 or self.api.port > 65535:
                logger.error("Invalid API port")
                return False
                
            # Validate trading settings
            if self.trading.risk_per_trade <= 0 or self.trading.risk_per_trade > 1:
                logger.error("Invalid risk per trade")
                return False
                
            # Validate ML settings
            if self.ml.validation_split <= 0 or self.ml.validation_split >= 1:
                logger.error("Invalid validation split")
                return False
                
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
            
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get current feature flags"""
        return {
            'trading_enabled': self.trading.trading_enabled,
            'paper_trading': self.trading.paper_trading,
            'real_time_updates': self.data.real_time_updates,
            'performance_monitoring': self.system.performance_monitoring,
            'compression_enabled': self.data.compression_enabled,
            'auto_save': self.ui.auto_save,
            'auto_refresh': self.ui.auto_refresh
        }
        
    def set_feature_flag(self, feature: str, enabled: bool):
        """Set feature flag"""
        if feature == 'trading_enabled':
            self.trading.trading_enabled = enabled
        elif feature == 'paper_trading':
            self.trading.paper_trading = enabled
        elif feature == 'real_time_updates':
            self.data.real_time_updates = enabled
        elif feature == 'performance_monitoring':
            self.system.performance_monitoring = enabled
        elif feature == 'compression_enabled':
            self.data.compression_enabled = enabled
        elif feature == 'auto_save':
            self.ui.auto_save = enabled
        elif feature == 'auto_refresh':
            self.ui.auto_refresh = enabled
        else:
            raise ValueError(f"Unknown feature flag: {feature}")
            
        logger.info(f"Feature flag '{feature}' set to {enabled}")

# Global configuration instance
config = MDPSConfig()

def get_config() -> MDPSConfig:
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload configuration from file"""
    config.load_config()

def save_config():
    """Save current configuration to file"""
    config.save_config()

if __name__ == "__main__":
    # Test configuration
    print("MDPS Configuration Test")
    print("=======================")
    
    # Validate configuration
    if config.validate_config():
        print("✓ Configuration validation passed")
    else:
        print("✗ Configuration validation failed")
        
    # Show feature flags
    print("\nFeature Flags:")
    for feature, enabled in config.get_feature_flags().items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature}")
        
    # Save configuration
    config.save_config()
    print(f"\nConfiguration saved to {config.config_file}")