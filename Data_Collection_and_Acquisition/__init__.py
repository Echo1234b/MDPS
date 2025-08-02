"""
Data Collection & Acquisition Module
Handles all data ingestion and initial processing tasks.
"""

from .data_connectivity_feed_integration.mt5_connection import MT5ConnectionManager

__all__ = ['MT5ConnectionManager']

class DataCollector:
    def __init__(self, config=None):
        self.mt5_connector = MetaTrader5Connector()
        self.candle_constructor = CandleConstructor()
        self.data_validator = DataValidator()
        self.data_storage = DataStorage()
        self.data_sanitizer = DataSanitizer()
        self.pipeline_orchestrator = PipelineOrchestrator()
        
    def initialize_feeds(self):
        """Initialize all data feeds"""
        return self.mt5_connector.connect()
        
    def collect_data(self, symbols, timeframe):
        """Collect and process data for given symbols"""
        raw_data = self.mt5_connector.get_data(symbols, timeframe)
        validated_data = self.data_validator.validate(raw_data)
        clean_data = self.data_sanitizer.clean(validated_data)
        return clean_data
