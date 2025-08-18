"""
Data Collection and Acquisition Module

This module handles all aspects of data collection and acquisition including:
- Multi-exchange API integration and WebSocket streaming
- MetaTrader 5 terminal integration
- Real-time data feeds with automatic failover
- Data validation and integrity assurance
- Time handling and candle construction
- Pipeline orchestration and monitoring

The module is designed for high-frequency trading data collection with
microsecond precision, comprehensive error handling, and scalable architecture.
"""

from .data_manager import DataCollectionManager, DataType, StorageFormat, DataRecord, DataQuery
from .validation import (
    DataCollectionValidator, ValidationResult, ValidationIssue, 
    ValidationLevel, ValidationStatus, ValidationRule
)

# Import submodules
from . import data_connectivity_feed_integration
from . import pre_cleaning_preparation  
from . import data_validation_integrity_assurance
from . import data_storage_profiling
from . import time_handling_candle_construction
from . import pipeline_orchestration_monitoring
from . import integration_protocols

__all__ = [
    # Main classes
    'DataCollectionManager',
    'DataCollectionValidator',
    
    # Data types and enums
    'DataType',
    'StorageFormat', 
    'DataRecord',
    'DataQuery',
    'ValidationResult',
    'ValidationIssue',
    'ValidationLevel',
    'ValidationStatus',
    'ValidationRule',
    
    # Submodules
    'data_connectivity_feed_integration',
    'pre_cleaning_preparation',
    'data_validation_integrity_assurance', 
    'data_storage_profiling',
    'time_handling_candle_construction',
    'pipeline_orchestration_monitoring',
    'integration_protocols'
]

__version__ = "1.0.0"
__author__ = "MDPS Development Team"
__description__ = "Comprehensive data collection and acquisition system for financial markets"

# Module-level configuration
DEFAULT_CONFIG = {
    'storage_path': './data/collection',
    'db_path': './data/collection.db',
    'max_memory_size': 1024 * 1024 * 1024,  # 1GB
    'validation_enabled': True,
    'compression_enabled': True,
    'backup_enabled': True,
    'monitoring_enabled': True
}

def get_default_config() -> dict:
    """Get default configuration for the module"""
    return DEFAULT_CONFIG.copy()

def create_data_manager(config: dict = None) -> DataCollectionManager:
    """
    Create a configured data manager instance
    
    Args:
        config: Configuration dictionary (uses defaults if None)
        
    Returns:
        DataCollectionManager: Configured data manager
    """
    if config is None:
        config = get_default_config()
    
    return DataCollectionManager(
        storage_path=config.get('storage_path', DEFAULT_CONFIG['storage_path']),
        db_path=config.get('db_path', DEFAULT_CONFIG['db_path']),
        max_memory_size=config.get('max_memory_size', DEFAULT_CONFIG['max_memory_size'])
    )

def create_validator(config: dict = None) -> DataCollectionValidator:
    """
    Create a configured validator instance
    
    Args:
        config: Configuration dictionary (uses defaults if None)
        
    Returns:
        DataCollectionValidator: Configured validator
    """
    validator = DataCollectionValidator()
    
    if config and not config.get('validation_enabled', True):
        # Remove all rules if validation is disabled
        validator.rules.clear()
    
    return validator

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"Initialized Data Collection and Acquisition module v{__version__}")