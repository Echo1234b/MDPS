"""
Data Input/Output Configuration and Management
"""
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta

class DataManager:
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "features",
            self.data_dir / "predictions",
            self.data_dir / "results"
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Configure logging"""
        log_file = self.data_dir / "mdps.log"
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def save_data(self, data, category, symbol, timeframe):
        """Save data to appropriate directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timeframe}_{timestamp}.csv"
        filepath = self.data_dir / category / filename
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath)
        else:
            pd.DataFrame(data).to_csv(filepath)
        
        logging.info(f"Saved {category} data to {filepath}")
        return filepath
        
    def load_data(self, category, symbol, timeframe, start_date=None, end_date=None):
        """Load data from storage"""
        directory = self.data_dir / category
        files = list(directory.glob(f"{symbol}_{timeframe}_*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No data found for {symbol} {timeframe} in {category}")
            
        # Load most recent file by default
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        data = pd.read_csv(latest_file)
        
        if start_date and end_date:
            data = data[(data['timestamp'] >= start_date) & 
                       (data['timestamp'] <= end_date)]
            
        return data
