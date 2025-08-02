"""
Market Data Processing System (MDPS) - Main Package Initialization
This file serves as the main entry point for the MDPS package.
"""

from pathlib import Path

# Define project root directory
PROJECT_ROOT = Path(__file__).parent

# Import core modules
from .Data_Collection_Acquisition import DataCollector
from .Data_Cleaning_Signal_Processing import DataCleaner
from .Preprocessing_Feature_Engineering import FeatureEngine
from .Advanced_Chart_Analysis_Tools import ChartAnalyzer
from .Market_Context_Structural_Analysis import MarketAnalyzer
from .External_Factors_Integration import ExternalFactors
from .Prediction_Engine import PredictionEngine
from .Strategy_Decision_Layer import StrategyManager

__version__ = "1.0.0"
