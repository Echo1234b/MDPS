"""
MDPS Main Entry Point
Demonstrates the integration and workflow of all MDPS components.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from . import (
    DataCollector,
    DataCleaner,
    FeatureEngine,
    ChartAnalyzer,
    MarketAnalyzer,
    ExternalFactors,
    PredictionEngine,
    StrategyManager
)
from .config import MDPSConfig

# Try to import data manager, create placeholder if not available
try:
    from .Data_Collection_and_Acquisition.data_manager import DataManager
except ImportError:
    logging.warning("DataManager not found - creating placeholder")
    class DataManager:
        def __init__(self, config):
            self.config = config
            self.data_dir = Path("data")
            self.data_dir.mkdir(exist_ok=True)
        
        def save_data(self, data, data_type, symbol, timeframe):
            logging.info(f"DataManager placeholder - saving {data_type} data for {symbol} {timeframe}")
            # Save to CSV for basic functionality
            filename = f"{symbol}_{timeframe}_{data_type}.csv"
            filepath = self.data_dir / filename
            try:
                data.to_csv(filepath, index=False)
                logging.info(f"Data saved to {filepath}")
            except Exception as e:
                logging.error(f"Error saving data: {e}")

# Try to import results handler, create placeholder if not available
try:
    from .Strategy_Decision_Layer.results_handler import ResultsHandler
except ImportError:
    logging.warning("ResultsHandler not found - creating placeholder")
    class ResultsHandler:
        def __init__(self, config):
            self.config = config
            self.results_dir = Path("results")
            self.results_dir.mkdir(exist_ok=True)
        
        def save_trading_signals(self, signals, symbol, timeframe):
            logging.info(f"ResultsHandler placeholder - saving signals for {symbol} {timeframe}")
            return True
        
        def generate_charts(self, data, patterns, predictions, symbol, timeframe):
            logging.info(f"ResultsHandler placeholder - generating charts for {symbol} {timeframe}")
            return True

import time
from datetime import datetime, timedelta

class MDPS:
    def __init__(self):
        # Initialize configuration
        self.config = MDPSConfig()
        
        # Initialize data management
        self.data_manager = DataManager(self.config)
        self.results_handler = ResultsHandler(self.config)
        
        # Initialize all components
        self.data_collector = DataCollector(self.config)
        self.data_cleaner = DataCleaner(self.config)
        self.feature_engine = FeatureEngine(self.config)
        self.chart_analyzer = ChartAnalyzer(self.config)
        self.market_analyzer = MarketAnalyzer(self.config)
        self.external_factors = ExternalFactors(self.config)
        self.prediction_engine = PredictionEngine(self.config)
        self.strategy_manager = StrategyManager(self.config)
        
    def initialize(self):
        """Initialize all system components"""
        try:
            logging.info("MDPS: Starting system initialization")
            
            # Start data collection
            self.data_collector.initialize_feeds()
            
            # Initialize analysis engines
            self.market_analyzer.initialize()
            self.external_factors.initialize()
            self.prediction_engine.load_models()
            
            # Start strategy manager
            self.strategy_manager.initialize()
            
            logging.info("MDPS: System initialized successfully")
        except Exception as e:
            logging.error(f"MDPS: Initialization error: {e}")
            raise
    
    def process_market_data(self, symbols, timeframe):
        """Main processing pipeline"""
        try:
            logging.info(f"MDPS: Starting processing pipeline for {symbols} at {timeframe}")
            
            # 1. Collect data
            raw_data = self.data_collector.collect_data(symbols, timeframe)
            self.data_manager.save_data(raw_data, "raw", symbols[0], timeframe)
            
            # 2. Clean and process signals
            clean_data = self.data_cleaner.process(raw_data)
            self.data_manager.save_data(clean_data, "processed", symbols[0], timeframe)
            
            # 3. Generate features
            features = self.feature_engine.generate_features(clean_data)
            self.data_manager.save_data(features, "features", symbols[0], timeframe)
            
            # 4. Perform technical analysis
            chart_patterns = self.chart_analyzer.analyze(clean_data)
            
            # 5. Analyze market structure
            market_context = self.market_analyzer.analyze_structure(clean_data)
            
            # 6. Integrate external factors
            external_data = self.external_factors.get_current_factors()
            
            # 7. Generate predictions
            predictions = self.prediction_engine.predict(
                features, 
                chart_patterns,
                market_context,
                external_data
            )
            
            # 8. Execute strategy decisions
            signals = self.strategy_manager.execute_decisions(
                predictions,
                market_context,
                external_data
            )
            
            # 9. Save results and generate reports
            self.results_handler.save_trading_signals(signals, symbols[0], timeframe)
            self.results_handler.generate_charts(clean_data, chart_patterns, predictions, symbols[0], timeframe)
            
            logging.info(f"MDPS: Processing pipeline completed successfully")
            
            # Return the processed results
            return {
                'signals': signals,
                'predictions': predictions,
                'market_context': market_context,
                'chart_patterns': chart_patterns
            }
            
        except Exception as e:
            logging.error(f"MDPS: Error in processing pipeline: {e}")
            raise
        
    def run(self, symbols, timeframe, update_interval=300):  # 5 minutes default
        """Run the entire system"""
        logging.info(f"MDPS: Starting main system loop")
        self.initialize()
        
        while True:
            try:
                start_time = time.time()
                
                # Process market data
                results = self.process_market_data(symbols, timeframe)
                
                # Log results summary
                logging.info(f"MDPS: Processed {symbols} at {timeframe} timeframe - Signal: {results['signals'].get('signal', 'none')}")
                
                # Calculate wait time for next update
                processing_time = time.time() - start_time
                wait_time = max(0, update_interval - processing_time)
                
                logging.info(f"MDPS: Processing took {processing_time:.2f}s, waiting {wait_time:.2f}s for next cycle")
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                logging.info("MDPS: Shutdown requested by user")
                break
            except Exception as e:
                logging.error(f"MDPS: Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mdps.log')
        ]
    )
    
    # Configuration
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframe = "M5"
    update_interval = 300  # 5 minutes
    
    logging.info("MDPS: Starting Market Data Processing System")
    
    # Create and run MDPS instance
    try:
        mdps = MDPS()
        mdps.run(symbols, timeframe, update_interval)
    except Exception as e:
        logging.critical(f"MDPS: System failure: {e}")
        raise

if __name__ == "__main__":
    main()
