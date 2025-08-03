#!/usr/bin/env python3
"""
MDPS Runner Script
Entry point for the Market Data Processing System
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import MDPS components
try:
    from main import MDPS, main as mdps_main
    from config import MDPSConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Attempting to import individual components...")
    
    # Try individual imports
    try:
        import __init__ as mdps_init
        from config import MDPSConfig
        
        # Use the placeholder classes from __init__.py
        DataCollector = mdps_init.DataCollector
        DataCleaner = mdps_init.DataCleaner
        FeatureEngine = mdps_init.FeatureEngine
        ChartAnalyzer = mdps_init.ChartAnalyzer
        MarketAnalyzer = mdps_init.MarketAnalyzer
        ExternalFactors = mdps_init.ExternalFactors
        PredictionEngine = mdps_init.PredictionEngine
        StrategyManager = mdps_init.StrategyManager
        
        # Create a simple MDPS class
        class MDPS:
            def __init__(self):
                self.config = MDPSConfig()
                self.data_collector = DataCollector(self.config)
                self.data_cleaner = DataCleaner(self.config)
                self.feature_engine = FeatureEngine(self.config)
                self.chart_analyzer = ChartAnalyzer(self.config)
                self.market_analyzer = MarketAnalyzer(self.config)
                self.external_factors = ExternalFactors(self.config)
                self.prediction_engine = PredictionEngine(self.config)
                self.strategy_manager = StrategyManager(self.config)
            
            def run_single_cycle(self, symbols, timeframe):
                """Run a single processing cycle"""
                try:
                    logging.info(f"MDPS: Running single cycle for {symbols} at {timeframe}")
                    
                    # 1. Collect data
                    raw_data = self.data_collector.collect_data(symbols, timeframe)
                    
                    # 2. Clean data
                    clean_data = self.data_cleaner.process(raw_data)
                    
                    # 3. Generate features
                    features = self.feature_engine.generate_features(clean_data)
                    
                    # 4. Analyze charts
                    chart_patterns = self.chart_analyzer.analyze(clean_data)
                    
                    # 5. Analyze market structure
                    market_context = self.market_analyzer.analyze_structure(clean_data)
                    
                    # 6. Get external factors
                    external_data = self.external_factors.get_current_factors()
                    
                    # 7. Generate predictions
                    predictions = self.prediction_engine.predict(
                        features, chart_patterns, market_context, external_data
                    )
                    
                    # 8. Execute strategy
                    signals = self.strategy_manager.execute_decisions(
                        predictions, market_context, external_data
                    )
                    
                    logging.info(f"MDPS: Cycle completed - Signal: {signals.get('signal', 'none')}")
                    
                    return {
                        'success': True,
                        'signals': signals,
                        'predictions': predictions,
                        'patterns': chart_patterns,
                        'market_context': market_context
                    }
                    
                except Exception as e:
                    logging.error(f"MDPS: Error in processing cycle: {e}")
                    return {'success': False, 'error': str(e)}
        
        def mdps_main():
            """Main entry point for MDPS"""
            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler('mdps.log')
                ]
            )
            
            logging.info("Starting Market Data Processing System")
            
            # Configuration
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            timeframe = "M5"
            
            try:
                mdps = MDPS()
                result = mdps.run_single_cycle(symbols, timeframe)
                
                if result['success']:
                    print(f"✅ MDPS cycle completed successfully!")
                    print(f"📊 Signal: {result['signals'].get('signal', 'none')}")
                    print(f"🔮 Prediction: {result['predictions'].get('direction', 'none')}")
                    print(f"📈 Patterns found: {len(result['patterns'].get('patterns', []))}")
                else:
                    print(f"❌ MDPS cycle failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logging.critical(f"MDPS: System failure: {e}")
                print(f"❌ System failure: {e}")
                
    except ImportError as e2:
        print(f"Failed to import components: {e2}")
        sys.exit(1)

def main():
    """Entry point"""
    print("🚀 Market Data Processing System (MDPS)")
    print("=" * 50)
    
    # Check if we can import everything properly
    try:
        mdps_main()
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        logging.error(f"Startup failed: {e}")

if __name__ == "__main__":
    main()
