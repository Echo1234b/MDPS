"""
MDPS Controller for PyQt UI Integration
Bridges the main MDPS system with the PyQt interface for real-time operation.
"""

import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import QMessageBox

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import MDPS
from config import MDPSConfig
from .event_system import EventSystem

class MDPSWorkerThread(QThread):
    """Worker thread for running MDPS processing without blocking UI"""
    
    data_processed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, mdps_instance, symbols, timeframe, update_interval):
        super().__init__()
        self.mdps = mdps_instance
        self.symbols = symbols
        self.timeframe = timeframe
        self.update_interval = update_interval
        self.running = False
        
    def run(self):
        """Main worker thread execution"""
        self.running = True
        self.status_update.emit("Initializing MDPS system...")
        
        try:
            # Initialize MDPS
            self.mdps.initialize()
            self.status_update.emit("MDPS system initialized successfully")
            
            while self.running:
                try:
                    start_time = time.time()
                    self.status_update.emit(f"Processing {self.symbols} data...")
                    
                    # Process market data
                    results = self.mdps.process_market_data(self.symbols, self.timeframe)
                    
                    # Emit processed data
                    self.data_processed.emit(results)
                    self.status_update.emit(f"Data processed successfully for {self.symbols}")
                    
                    # Calculate wait time for next update
                    processing_time = time.time() - start_time
                    wait_time = max(0, self.update_interval - processing_time)
                    
                    # Sleep with interruption check
                    sleep_start = time.time()
                    while time.time() - sleep_start < wait_time and self.running:
                        self.msleep(100)  # Sleep 100ms at a time
                        
                except Exception as e:
                    self.error_occurred.emit(f"Processing error: {str(e)}")
                    self.msleep(5000)  # Wait 5 seconds before retry
                    
        except Exception as e:
            self.error_occurred.emit(f"Initialization error: {str(e)}")
            
    def stop(self):
        """Stop the worker thread"""
        self.running = False

class MDPSController(QObject):
    """Controller class to manage MDPS system integration with PyQt UI"""
    
    # Signals for UI updates
    data_updated = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    connection_status_changed = pyqtSignal(bool)
    
    def __init__(self, config: MDPSConfig = None, event_system: EventSystem = None):
        super().__init__()
        self.config = config or MDPSConfig()
        self.event_system = event_system or EventSystem()
        self.mdps = None
        self.worker_thread = None
        self.is_running = False
        
        # Default settings
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        self.timeframe = "M5"
        self.update_interval = 300  # 5 minutes
        
        # Setup event connections
        self.setup_event_connections()
        
    def setup_event_connections(self):
        """Setup connections between MDPS events and UI signals"""
        if self.event_system:
            # Connect internal events to UI signals
            self.event_system.subscribe('data_processed', self.on_data_processed)
            self.event_system.subscribe('error_occurred', self.on_error_occurred)
            self.event_system.subscribe('status_update', self.on_status_update)
    
    def initialize_mdps(self) -> bool:
        """Initialize MDPS system"""
        try:
            self.status_changed.emit("Initializing MDPS...")
            self.mdps = MDPS()
            self.status_changed.emit("MDPS system ready")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize MDPS: {str(e)}")
            return False
    
    def start_processing(self, symbols: list = None, timeframe: str = None, 
                        update_interval: int = None) -> bool:
        """Start MDPS processing in background thread"""
        if self.is_running:
            self.error_occurred.emit("MDPS is already running")
            return False
            
        if not self.mdps:
            if not self.initialize_mdps():
                return False
        
        # Update settings
        if symbols:
            self.symbols = symbols
        if timeframe:
            self.timeframe = timeframe
        if update_interval:
            self.update_interval = update_interval
            
        try:
            # Create and start worker thread
            self.worker_thread = MDPSWorkerThread(
                self.mdps, self.symbols, self.timeframe, self.update_interval
            )
            
            # Connect worker signals
            self.worker_thread.data_processed.connect(self.on_data_processed)
            self.worker_thread.error_occurred.connect(self.on_error_occurred)
            self.worker_thread.status_update.connect(self.on_status_update)
            
            # Start worker
            self.worker_thread.start()
            self.is_running = True
            self.connection_status_changed.emit(True)
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start processing: {str(e)}")
            return False
    
    def stop_processing(self):
        """Stop MDPS processing"""
        if self.worker_thread and self.is_running:
            self.status_changed.emit("Stopping MDPS processing...")
            self.worker_thread.stop()
            self.worker_thread.wait(5000)  # Wait up to 5 seconds
            self.worker_thread = None
            self.is_running = False
            self.connection_status_changed.emit(False)
            self.status_changed.emit("MDPS processing stopped")
    
    def on_data_processed(self, results: dict):
        """Handle processed data from MDPS"""
        # Emit to UI components
        self.data_updated.emit(results)
        
        # Publish to event system
        if self.event_system:
            self.event_system.publish('mdps_data_updated', results)
    
    def on_error_occurred(self, error_message: str):
        """Handle errors from MDPS processing"""
        self.error_occurred.emit(error_message)
        
        # Publish to event system
        if self.event_system:
            self.event_system.publish('mdps_error', error_message)
    
    def on_status_update(self, status: str):
        """Handle status updates from MDPS processing"""
        self.status_changed.emit(status)
        
        # Publish to event system
        if self.event_system:
            self.event_system.publish('mdps_status', status)
    
    def get_current_results(self) -> Optional[Dict[str, Any]]:
        """Get the last processed results"""
        if self.mdps and hasattr(self.mdps, 'last_results'):
            return self.mdps.last_results
        return None
    
    def update_symbols(self, symbols: list):
        """Update the symbols being processed"""
        self.symbols = symbols
        if self.is_running:
            # Restart processing with new symbols
            self.stop_processing()
            self.start_processing()
    
    def update_timeframe(self, timeframe: str):
        """Update the timeframe for processing"""
        self.timeframe = timeframe
        if self.is_running:
            # Restart processing with new timeframe
            self.stop_processing()
            self.start_processing()
    
    def update_interval(self, interval: int):
        """Update the processing interval"""
        self.update_interval = interval
        
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'update_interval': self.update_interval,
            'worker_active': self.worker_thread is not None and self.worker_thread.isRunning()
        }