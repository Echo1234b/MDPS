"""
External Factors Integration Module

This module handles integration of external factors that influence market behavior:
- News and economic events analysis
- Social media and crypto sentiment tracking
- Market microstructure and correlation analysis
- Blockchain and on-chain analytics
- Time-weighted event impact modeling

The module provides comprehensive external data integration with real-time processing,
sentiment analysis, and impact assessment capabilities.
"""

from .data_manager import ExternalFactorsDataManager
from .validation import ExternalFactorsValidator
from .api_interface import ExternalFactorsAPIInterface
from .event_bus import ExternalFactorsEventBus

# Import submodules
from . import NewsAndEconomicEvents
from . import SocialAndCryptoSentiment
from . import MarketMicrostructureAndCorrelations
from . import BlockchainAndOnChainAnalytics
from . import TimeWeightedEventImpactModel
from . import integration_protocols

__all__ = [
    # Main classes
    'ExternalFactorsDataManager',
    'ExternalFactorsValidator', 
    'ExternalFactorsAPIInterface',
    'ExternalFactorsEventBus',
    
    # Submodules
    'NewsAndEconomicEvents',
    'SocialAndCryptoSentiment',
    'MarketMicrostructureAndCorrelations',
    'BlockchainAndOnChainAnalytics', 
    'TimeWeightedEventImpactModel',
    'integration_protocols'
]

__version__ = "1.0.0"
__author__ = "MDPS Development Team"
__description__ = "External factors integration system for comprehensive market analysis"