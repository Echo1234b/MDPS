#!/usr/bin/env python3
"""
MDPS Logging Management
Centralized logging with rotation, formatting, and multiple output handlers
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import traceback

class MDPSLogger:
    """Centralized logging manager for MDPS system"""
    
    def __init__(self, name: str = "MDPS", level: str = "INFO", 
                 log_file: str = "mdps.log", max_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 5):
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_file = log_file
        self.max_size = max_size
        self.backup_count = backup_count
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup the main logger"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        json_formatter = JSONFormatter()
        
        # Console handler (simple format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # File handler (detailed format)
        file_handler = logging.handlers.RotatingFileHandler(
            f"logs/{self.log_file}",
            maxBytes=self.max_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler (errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            f"logs/errors_{self.log_file}",
            maxBytes=self.max_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # JSON handler for structured logging
        json_handler = logging.handlers.RotatingFileHandler(
            f"logs/structured_{self.log_file}",
            maxBytes=self.max_size,
            backupCount=self.backup_count
        )
        json_handler.setLevel(self.level)
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)
        
        return logger
        
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance"""
        if name:
            return logging.getLogger(f"{self.name}.{name}")
        return self.logger
        
    def set_level(self, level: str):
        """Set logging level"""
        new_level = getattr(logging, level.upper())
        self.logger.setLevel(new_level)
        
        # Update all handlers
        for handler in self.logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                if "errors_" in handler.baseFilename:
                    handler.setLevel(logging.ERROR)
                else:
                    handler.setLevel(new_level)
                    
    def add_file_handler(self, filename: str, level: str = "INFO", 
                        formatter: str = "detailed"):
        """Add additional file handler"""
        if formatter == "detailed":
            formatter_obj = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
            )
        elif formatter == "simple":
            formatter_obj = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        elif formatter == "json":
            formatter_obj = JSONFormatter()
        else:
            formatter_obj = logging.Formatter('%(message)s')
            
        handler = logging.handlers.RotatingFileHandler(
            f"logs/{filename}",
            maxBytes=self.max_size,
            backupCount=self.backup_count
        )
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(formatter_obj)
        
        self.logger.addHandler(handler)
        
    def remove_handler(self, handler):
        """Remove a specific handler"""
        if handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            handler.close()
            
    def log_performance(self, operation: str, duration: float, 
                       details: Dict[str, Any] = None):
        """Log performance metrics"""
        message = f"PERFORMANCE: {operation} took {duration:.4f}s"
        if details:
            message += f" - {json.dumps(details)}"
            
        self.logger.info(message)
        
    def log_error_with_context(self, error: Exception, context: str = "", 
                             additional_data: Dict[str, Any] = None):
        """Log error with context and additional data"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_data:
            error_info.update(additional_data)
            
        self.logger.error(f"ERROR in {context}: {str(error)}")
        self.logger.error(f"Error details: {json.dumps(error_info, indent=2)}")
        
    def log_system_event(self, event_type: str, event_data: Dict[str, Any] = None):
        """Log system events"""
        message = f"SYSTEM_EVENT: {event_type}"
        if event_data:
            message += f" - {json.dumps(event_data)}"
            
        self.logger.info(message)
        
    def log_trading_event(self, symbol: str, event_type: str, 
                         event_data: Dict[str, Any] = None):
        """Log trading events"""
        message = f"TRADING_EVENT: {symbol} - {event_type}"
        if event_data:
            message += f" - {json.dumps(event_data)}"
            
        self.logger.info(message)
        
    def log_data_event(self, data_type: str, symbol: str = None, 
                      event_data: Dict[str, Any] = None):
        """Log data processing events"""
        message = f"DATA_EVENT: {data_type}"
        if symbol:
            message += f" - {symbol}"
        if event_data:
            message += f" - {json.dumps(event_data)}"
            
        self.logger.info(message)
        
    def log_ml_event(self, model_name: str, event_type: str, 
                    event_data: Dict[str, Any] = None):
        """Log machine learning events"""
        message = f"ML_EVENT: {model_name} - {event_type}"
        if event_data:
            message += f" - {json.dumps(event_data)}"
            
        self.logger.info(message)
        
    def get_log_files(self) -> Dict[str, str]:
        """Get list of log files"""
        log_dir = Path("logs")
        log_files = {}
        
        if log_dir.exists():
            for file in log_dir.glob("*.log*"):
                log_files[file.name] = str(file)
                
        return log_files
        
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            'total_handlers': len(self.logger.handlers),
            'log_level': logging.getLevelName(self.logger.level),
            'log_files': self.get_log_files(),
            'handler_types': []
        }
        
        for handler in self.logger.handlers:
            stats['handler_types'].append(type(handler).__name__)
            
        return stats
        
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        try:
            log_dir = Path("logs")
            if not log_dir.exists():
                return
                
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            for file in log_dir.glob("*.log*"):
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()
                    self.logger.info(f"Removed old log file: {file}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry)

class PerformanceLogger:
    """Performance logging decorator and context manager"""
    
    def __init__(self, logger: MDPSLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.log_performance(self.operation, duration)
            
    def __call__(self, func):
        """Decorator to log function performance"""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.log_performance(f"{self.operation}.{func.__name__}", duration)
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.log_performance(f"{self.operation}.{func.__name__}", duration, 
                                          {'error': str(e)})
                raise
        return wrapper

def setup_logging(level: str = "INFO", log_file: str = "mdps.log", 
                 max_size: int = 100 * 1024 * 1024, backup_count: int = 5) -> MDPSLogger:
    """Setup global logging"""
    global_logger = MDPSLogger(
        name="MDPS",
        level=level,
        log_file=log_file,
        max_size=max_size,
        backup_count=backup_count
    )
    
    return global_logger

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance"""
    # This would get the global logger instance
    # For now, create a new one if needed
    if not hasattr(get_logger, '_global_logger'):
        get_logger._global_logger = MDPSLogger()
        
    return get_logger._global_logger.get_logger(name)

def log_performance(operation: str):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger = get_logger()
                logger.info(f"PERFORMANCE: {operation}.{func.__name__} took {duration:.4f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger = get_logger()
                logger.error(f"PERFORMANCE: {operation}.{func.__name__} took {duration:.4f}s (ERROR: {e})")
                raise
        return wrapper
    return decorator

def log_errors(context: str = ""):
    """Decorator to log errors with context"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger()
                logger.log_error_with_context(e, context)
                raise
        return wrapper
    return decorator

# Global logger instance
_global_logger = None

def get_global_logger() -> MDPSLogger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = MDPSLogger()
    return _global_logger

if __name__ == "__main__":
    # Test logging functionality
    logger = setup_logging(level="DEBUG")
    
    # Test different log levels
    logger.logger.debug("Debug message")
    logger.logger.info("Info message")
    logger.logger.warning("Warning message")
    logger.logger.error("Error message")
    
    # Test performance logging
    with PerformanceLogger(logger, "test_operation"):
        import time
        time.sleep(0.1)
        
    # Test structured logging
    logger.log_system_event("startup", {"version": "1.0.0", "timestamp": datetime.now().isoformat()})
    logger.log_trading_event("BTCUSD", "signal_generated", {"signal_type": "buy", "strength": 0.8})
    logger.log_data_event("market_data_received", "BTCUSD", {"timeframe": "1m", "records": 100})
    logger.log_ml_event("lstm_model", "prediction_made", {"prediction": 0.75, "confidence": 0.85})
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error_with_context(e, "test_function", {"test_data": "example"})
        
    # Show log stats
    stats = logger.get_log_stats()
    print("Log stats:", json.dumps(stats, indent=2))
    
    # Show log files
    log_files = logger.get_log_files()
    print("Log files:", log_files)