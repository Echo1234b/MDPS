#!/usr/bin/env python3
"""
MDPS System Main Entry Point
Enhanced system initialization with orchestration, module dependency resolution, and system health checks
"""

import os
import sys
import time
import signal
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import traceback

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, config
from database import DatabaseManager
from logging import setup_logging
from error_handling import ErrorHandler
from orchestrator import SystemOrchestrator

# Configure logging
logger = logging.getLogger(__name__)

class MDPSSystem:
    """Main MDPS system class"""
    
    def __init__(self):
        self.config = get_config()
        self.error_handler = ErrorHandler()
        self.database = None
        self.orchestrator = None
        self.modules = {}
        self.running = False
        self.startup_time = None
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup system signal handlers"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        
    async def initialize(self):
        """Initialize the MDPS system"""
        try:
            logger.info("Initializing MDPS System...")
            self.startup_time = time.time()
            
            # Validate configuration
            if not self.config.validate_config():
                logger.error("Configuration validation failed")
                return False
                
            # Setup logging
            setup_logging(
                level=self.config.system.log_level,
                log_file=self.config.system.log_file,
                max_size=self.config.system.max_log_size,
                backup_count=self.config.system.log_backup_count
            )
            
            # Initialize database
            await self._initialize_database()
            
            # Initialize orchestrator
            await self._initialize_orchestrator()
            
            # Initialize system modules
            await self._initialize_modules()
            
            # Perform system health checks
            if not await self._perform_health_checks():
                logger.error("System health checks failed")
                return False
                
            # Initialize user interface
            await self._initialize_ui()
            
            # Prepare backup and recovery
            await self._prepare_backup_recovery()
            
            logger.info("MDPS System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            traceback.print_exc()
            return False
            
    async def _initialize_database(self):
        """Initialize database connection"""
        try:
            logger.info("Initializing database...")
            self.database = DatabaseManager(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                username=self.config.database.username,
                password=self.config.database.password
            )
            await self.database.connect()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
            
    async def _initialize_orchestrator(self):
        """Initialize system orchestrator"""
        try:
            logger.info("Initializing system orchestrator...")
            self.orchestrator = SystemOrchestrator(
                config=self.config,
                database=self.database,
                error_handler=self.error_handler
            )
            await self.orchestrator.initialize()
            logger.info("System orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            raise
            
    async def _initialize_modules(self):
        """Initialize system modules"""
        try:
            logger.info("Initializing system modules...")
            
            # Initialize data collection module
            await self._initialize_data_collection()
            
            # Initialize data processing modules
            await self._initialize_data_processing()
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Initialize strategy layer
            await self._initialize_strategy_layer()
            
            # Initialize UI framework
            await self._initialize_ui_framework()
            
            logger.info("All system modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Module initialization failed: {e}")
            raise
            
    async def _initialize_data_collection(self):
        """Initialize data collection modules"""
        try:
            logger.info("Initializing data collection modules...")
            
            # Data Collection and Acquisition
            from "1. Data_Collection_and_Acquisition".data_manager import DataCollectionManager
            self.modules['data_collection'] = DataCollectionManager(
                config=self.config,
                database=self.database
            )
            await self.modules['data_collection'].initialize()
            
            # External Factors Integration
            from "2. External Factors Integration".data_manager import ExternalFactorsManager
            self.modules['external_factors'] = ExternalFactorsManager(
                config=self.config,
                database=self.database
            )
            await self.modules['external_factors'].initialize()
            
            logger.info("Data collection modules initialized")
            
        except Exception as e:
            logger.error(f"Data collection initialization failed: {e}")
            raise
            
    async def _initialize_data_processing(self):
        """Initialize data processing modules"""
        try:
            logger.info("Initializing data processing modules...")
            
            # Data Cleaning & Signal Processing
            from "3. Data Cleaning & Signal Processing".data_manager import DataCleaningManager
            self.modules['data_cleaning'] = DataCleaningManager(
                config=self.config,
                database=self.database
            )
            await self.modules['data_cleaning'].initialize()
            
            # Preprocessing & Feature Engineering
            from "4. Preprocessing & Feature Engineering".data_manager import FeatureEngineeringManager
            self.modules['feature_engineering'] = FeatureEngineeringManager(
                config=self.config,
                database=self.config
            )
            await self.modules['feature_engineering'].initialize()
            
            # Market Context Analysis
            from "5. Market_Context_Structural_Analysis".data_manager import MarketContextManager
            self.modules['market_context'] = MarketContextManager(
                config=self.config,
                database=self.database
            )
            await self.modules['market_context'].initialize()
            
            logger.info("Data processing modules initialized")
            
        except Exception as e:
            logger.error(f"Data processing initialization failed: {e}")
            raise
            
    async def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            logger.info("Initializing ML models...")
            
            # Prediction Engine
            from "8. Prediction Engine (MLDL Models)".data_manager import PredictionEngineManager
            self.modules['prediction_engine'] = PredictionEngineManager(
                config=self.config,
                database=self.database
            )
            await self.modules['prediction_engine'].initialize()
            
            # Labeling & Target Engineering
            from "7. Labeling & Target Engineering".data_manager import LabelingManager
            self.modules['labeling'] = LabelingManager(
                config=self.config,
                database=self.database
            )
            await self.modules['labeling'].initialize()
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
            raise
            
    async def _initialize_strategy_layer(self):
        """Initialize strategy and decision layer"""
        try:
            logger.info("Initializing strategy layer...")
            
            from "9. Strategy & Decision Layer".data_manager import StrategyManager
            self.modules['strategy'] = StrategyManager(
                config=self.config,
                database=self.database
            )
            await self.modules['strategy'].initialize()
            
            logger.info("Strategy layer initialized")
            
        except Exception as e:
            logger.error(f"Strategy layer initialization failed: {e}")
            raise
            
    async def _initialize_ui_framework(self):
        """Initialize UI framework"""
        try:
            logger.info("Initializing UI framework...")
            
            from "10. trading_ui".core.mdps_controller import MDPSController
            self.modules['ui_controller'] = MDPSController(
                config=self.config,
                orchestrator=self.orchestrator
            )
            await self.modules['ui_controller'].initialize()
            
            logger.info("UI framework initialized")
            
        except Exception as e:
            logger.error(f"UI framework initialization failed: {e}")
            raise
            
    async def _perform_health_checks(self):
        """Perform system health checks"""
        try:
            logger.info("Performing system health checks...")
            
            # Check database connectivity
            if not await self.database.health_check():
                logger.error("Database health check failed")
                return False
                
            # Check module health
            for name, module in self.modules.items():
                if hasattr(module, 'health_check'):
                    if not await module.health_check():
                        logger.error(f"Module {name} health check failed")
                        return False
                        
            # Check system resources
            if not await self._check_system_resources():
                logger.error("System resource check failed")
                return False
                
            logger.info("All health checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Health checks failed: {e}")
            return False
            
    async def _check_system_resources(self):
        """Check system resources"""
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")
                
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
                
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                logger.warning(f"High disk usage: {disk.percent}%")
                
            return True
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False
            
    async def _initialize_ui(self):
        """Initialize user interface"""
        try:
            logger.info("Initializing user interface...")
            
            # Initialize main UI window
            from "10. trading_ui".ui.main_window import MainWindow
            self.ui = MainWindow(
                config=self.config,
                orchestrator=self.orchestrator
            )
            await self.ui.initialize()
            
            logger.info("User interface initialized")
            
        except Exception as e:
            logger.error(f"UI initialization failed: {e}")
            raise
            
    async def _prepare_backup_recovery(self):
        """Prepare backup and recovery systems"""
        try:
            logger.info("Preparing backup and recovery systems...")
            
            # Create backup directories
            backup_dir = Path(self.config.data.backup_dir)
            backup_dir.mkdir(exist_ok=True)
            
            # Initialize backup manager
            from backup_manager import BackupManager
            self.backup_manager = BackupManager(
                config=self.config,
                database=self.database
            )
            await self.backup_manager.initialize()
            
            logger.info("Backup and recovery systems prepared")
            
        except Exception as e:
            logger.error(f"Backup preparation failed: {e}")
            raise
            
    async def start(self):
        """Start the MDPS system"""
        try:
            logger.info("Starting MDPS System...")
            
            # Start orchestrator
            await self.orchestrator.start()
            
            # Start all modules
            for name, module in self.modules.items():
                if hasattr(module, 'start'):
                    await module.start()
                    logger.info(f"Module {name} started")
                    
            # Start UI
            await self.ui.start()
            
            self.running = True
            logger.info("MDPS System started successfully")
            
            # Start monitoring
            await self._start_monitoring()
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            await self.shutdown()
            raise
            
    async def _start_monitoring(self):
        """Start system monitoring"""
        try:
            logger.info("Starting system monitoring...")
            
            # Start performance monitoring
            if self.config.system.performance_monitoring:
                await self._start_performance_monitoring()
                
            # Start health monitoring
            await self._start_health_monitoring()
            
            logger.info("System monitoring started")
            
        except Exception as e:
            logger.error(f"Monitoring startup failed: {e}")
            
    async def _start_performance_monitoring(self):
        """Start performance monitoring"""
        async def monitor_performance():
            while self.running:
                try:
                    # Collect performance metrics
                    metrics = await self._collect_performance_metrics()
                    
                    # Store metrics
                    await self.database.store_metrics(metrics)
                    
                    # Check for performance issues
                    await self._check_performance_alerts(metrics)
                    
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    await asyncio.sleep(60)
                    
        asyncio.create_task(monitor_performance())
        
    async def _start_health_monitoring(self):
        """Start health monitoring"""
        async def monitor_health():
            while self.running:
                try:
                    # Perform health checks
                    if not await self._perform_health_checks():
                        logger.error("System health check failed")
                        await self._handle_health_failure()
                        
                    await asyncio.sleep(self.config.system.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(60)
                    
        asyncio.create_task(monitor_health())
        
    async def _collect_performance_metrics(self):
        """Collect system performance metrics"""
        try:
            import psutil
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters()._asdict(),
                'module_status': {}
            }
            
            # Collect module-specific metrics
            for name, module in self.modules.items():
                if hasattr(module, 'get_metrics'):
                    metrics['module_status'][name] = await module.get_metrics()
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}
            
    async def _check_performance_alerts(self, metrics):
        """Check for performance alerts"""
        try:
            # CPU alert
            if metrics.get('cpu_percent', 0) > 90:
                await self._send_alert('High CPU usage detected', 'WARNING')
                
            # Memory alert
            if metrics.get('memory_percent', 0) > 90:
                await self._send_alert('High memory usage detected', 'WARNING')
                
            # Disk alert
            if metrics.get('disk_percent', 0) > 90:
                await self._send_alert('High disk usage detected', 'WARNING')
                
        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")
            
    async def _send_alert(self, message: str, level: str):
        """Send system alert"""
        try:
            logger.warning(f"ALERT [{level}]: {message}")
            
            # Store alert in database
            await self.database.store_alert(message, level)
            
            # Send to orchestrator
            if self.orchestrator:
                await self.orchestrator.handle_alert(message, level)
                
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            
    async def _handle_health_failure(self):
        """Handle system health failure"""
        try:
            logger.error("System health failure detected, attempting recovery...")
            
            # Attempt automatic recovery
            if await self._attempt_recovery():
                logger.info("System recovery successful")
            else:
                logger.error("System recovery failed")
                await self.shutdown()
                
        except Exception as e:
            logger.error(f"Health failure handling failed: {e}")
            
    async def _attempt_recovery(self):
        """Attempt system recovery"""
        try:
            # Restart failed modules
            for name, module in self.modules.items():
                if hasattr(module, 'health_check'):
                    if not await module.health_check():
                        logger.info(f"Attempting to restart module {name}")
                        if hasattr(module, 'restart'):
                            await module.restart()
                            
            # Wait for recovery
            await asyncio.sleep(10)
            
            # Check if recovery was successful
            return await self._perform_health_checks()
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the MDPS system"""
        try:
            logger.info("Shutting down MDPS System...")
            self.running = False
            
            # Shutdown UI
            if hasattr(self, 'ui'):
                await self.ui.shutdown()
                
            # Shutdown modules
            for name, module in self.modules.items():
                if hasattr(module, 'shutdown'):
                    await module.shutdown()
                    logger.info(f"Module {name} shut down")
                    
            # Shutdown orchestrator
            if self.orchestrator:
                await self.orchestrator.shutdown()
                
            # Shutdown database
            if self.database:
                await self.database.disconnect()
                
            # Perform cleanup
            await self._cleanup()
            
            logger.info("MDPS System shutdown completed")
            
        except Exception as e:
            logger.error(f"System shutdown failed: {e}")
            
    async def _cleanup(self):
        """Perform system cleanup"""
        try:
            logger.info("Performing system cleanup...")
            
            # Create backup before shutdown
            if hasattr(self, 'backup_manager'):
                await self.backup_manager.create_backup()
                
            # Clean up temporary files
            await self._cleanup_temp_files()
            
            # Log shutdown statistics
            if self.startup_time:
                uptime = time.time() - self.startup_time
                logger.info(f"System uptime: {uptime:.2f} seconds")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            
    async def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import tempfile
            import shutil
            
            # Clean up system temp directory
            temp_dir = tempfile.gettempdir()
            mdps_temp = os.path.join(temp_dir, 'mdps')
            if os.path.exists(mdps_temp):
                shutil.rmtree(mdps_temp)
                
            logger.info("Temporary files cleaned up")
            
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")
            
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            'running': self.running,
            'startup_time': self.startup_time,
            'uptime': time.time() - self.startup_time if self.startup_time else 0,
            'modules': list(self.modules.keys()),
            'config': {
                'trading_enabled': self.config.trading.trading_enabled,
                'paper_trading': self.config.trading.paper_trading,
                'real_time_updates': self.config.data.real_time_updates
            }
        }

async def main():
    """Main entry point"""
    try:
        # Create and initialize system
        system = MDPSSystem()
        
        # Initialize system
        if not await system.initialize():
            logger.error("System initialization failed")
            sys.exit(1)
            
        # Start system
        await system.start()
        
        # Keep system running
        try:
            while system.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await system.shutdown()
            
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())