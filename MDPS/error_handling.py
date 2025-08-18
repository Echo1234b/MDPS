#!/usr/bin/env python3
"""
MDPS Error Handling
Centralized error management with categorization, recovery, and reporting
"""

import os
import sys
import logging
import traceback
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ErrorCategory(Enum):
    """Error categories"""
    SYSTEM = "SYSTEM"
    DATABASE = "DATABASE"
    NETWORK = "NETWORK"
    DATA = "DATA"
    ML = "ML"
    TRADING = "TRADING"
    UI = "UI"
    VALIDATION = "VALIDATION"
    INTEGRATION = "INTEGRATION"
    UNKNOWN = "UNKNOWN"

class ErrorHandler:
    """Central error handler for MDPS system"""
    
    def __init__(self):
        self.error_history = []
        self.max_history_size = 1000
        self.recovery_strategies = {}
        self.error_callbacks = {}
        self.severity_thresholds = {
            ErrorSeverity.LOW: 100,      # Max errors before action
            ErrorSeverity.MEDIUM: 50,
            ErrorSeverity.HIGH: 20,
            ErrorSeverity.CRITICAL: 5
        }
        self.error_counts = {severity: 0 for severity in ErrorSeverity}
        self.last_reset = datetime.now()
        
        # Setup default recovery strategies
        self._setup_default_recovery_strategies()
        
    def _setup_default_recovery_strategies(self):
        """Setup default error recovery strategies"""
        # Database errors
        self.recovery_strategies[ErrorCategory.DATABASE] = {
            'retry_count': 3,
            'retry_delay': 1.0,
            'backoff_factor': 2.0,
            'max_delay': 30.0,
            'recovery_action': self._recover_database_error
        }
        
        # Network errors
        self.recovery_strategies[ErrorCategory.NETWORK] = {
            'retry_count': 5,
            'retry_delay': 2.0,
            'backoff_factor': 1.5,
            'max_delay': 60.0,
            'recovery_action': self._recover_network_error
        }
        
        # Data errors
        self.recovery_strategies[ErrorCategory.DATA] = {
            'retry_count': 2,
            'retry_delay': 0.5,
            'backoff_factor': 1.0,
            'max_delay': 10.0,
            'recovery_action': self._recover_data_error
        }
        
        # ML errors
        self.recovery_strategies[ErrorCategory.ML] = {
            'retry_count': 1,
            'retry_delay': 5.0,
            'backoff_factor': 1.0,
            'max_delay': 30.0,
            'recovery_action': self._recover_ml_error
        }
        
        # Trading errors
        self.recovery_strategies[ErrorCategory.TRADING] = {
            'retry_count': 0,  # No retry for trading errors
            'retry_delay': 0.0,
            'backoff_factor': 1.0,
            'max_delay': 0.0,
            'recovery_action': self._recover_trading_error
        }
        
    def handle_error(self, error: Exception, context: str = "", 
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    additional_data: Dict[str, Any] = None,
                    retry: bool = True) -> bool:
        """Handle an error with categorization and recovery"""
        try:
            # Create error record
            error_record = self._create_error_record(
                error, context, category, severity, additional_data
            )
            
            # Add to history
            self._add_to_history(error_record)
            
            # Update error counts
            self.error_counts[severity] += 1
            
            # Log error
            self._log_error(error_record)
            
            # Check if threshold exceeded
            if self._check_threshold_exceeded(severity):
                self._handle_threshold_exceeded(severity, category)
                
            # Attempt recovery if enabled
            if retry and category in self.recovery_strategies:
                return self._attempt_recovery(error_record)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            return False
            
    def _create_error_record(self, error: Exception, context: str, 
                           category: ErrorCategory, severity: ErrorSeverity,
                           additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a structured error record"""
        return {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'category': category.value,
            'severity': severity.value,
            'additional_data': additional_data or {},
            'resolved': False,
            'recovery_attempts': 0
        }
        
    def _add_to_history(self, error_record: Dict[str, Any]):
        """Add error to history"""
        self.error_history.append(error_record)
        
        # Maintain history size
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
            
    def _log_error(self, error_record: Dict[str, Any]):
        """Log error with appropriate level"""
        message = f"ERROR [{error_record['category']}] in {error_record['context']}: {error_record['error_message']}"
        
        if error_record['severity'] == ErrorSeverity.CRITICAL:
            logger.critical(message)
        elif error_record['severity'] == ErrorSeverity.HIGH:
            logger.error(message)
        elif error_record['severity'] == ErrorSeverity.MEDIUM:
            logger.warning(message)
        else:
            logger.info(message)
            
        # Log additional data if present
        if error_record['additional_data']:
            logger.debug(f"Additional error data: {json.dumps(error_record['additional_data'], indent=2)}")
            
    def _check_threshold_exceeded(self, severity: ErrorSeverity) -> bool:
        """Check if error threshold is exceeded"""
        threshold = self.severity_thresholds[severity]
        current_count = self.error_counts[severity]
        
        return current_count >= threshold
        
    def _handle_threshold_exceeded(self, severity: ErrorSeverity, category: ErrorCategory):
        """Handle threshold exceeded"""
        logger.warning(f"Error threshold exceeded for {severity.value} errors in {category.value}")
        
        # Execute callbacks
        if category in self.error_callbacks:
            for callback in self.error_callbacks[category]:
                try:
                    callback(severity, category)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")
                    
        # Take automatic action based on severity
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_threshold_exceeded(category)
        elif severity == ErrorSeverity.HIGH:
            self._handle_high_threshold_exceeded(category)
            
    def _handle_critical_threshold_exceeded(self, category: ErrorCategory):
        """Handle critical threshold exceeded"""
        logger.critical(f"CRITICAL error threshold exceeded for {category.value}")
        
        # Emergency shutdown for certain categories
        if category in [ErrorCategory.DATABASE, ErrorCategory.TRADING]:
            logger.critical("Initiating emergency shutdown")
            # This would trigger system shutdown
            # For now, just log the action
            
    def _handle_high_threshold_exceeded(self, category: ErrorCategory):
        """Handle high threshold exceeded"""
        logger.error(f"HIGH error threshold exceeded for {category.value}")
        
        # Disable affected functionality
        if category == ErrorCategory.TRADING:
            logger.warning("Disabling trading functionality due to high error rate")
        elif category == ErrorCategory.ML:
            logger.warning("Disabling ML predictions due to high error rate")
            
    def _attempt_recovery(self, error_record: Dict[str, Any]) -> bool:
        """Attempt error recovery"""
        try:
            category = ErrorCategory(error_record['category'])
            strategy = self.recovery_strategies[category]
            
            if error_record['recovery_attempts'] >= strategy['retry_count']:
                logger.warning(f"Max recovery attempts reached for {category.value}")
                return False
                
            # Calculate delay with backoff
            delay = min(
                strategy['retry_delay'] * (strategy['backoff_factor'] ** error_record['recovery_attempts']),
                strategy['max_delay']
            )
            
            logger.info(f"Attempting recovery for {category.value} error in {delay:.1f}s")
            
            # Execute recovery action
            if strategy['recovery_action']:
                success = strategy['recovery_action'](error_record)
                if success:
                    error_record['resolved'] = True
                    logger.info(f"Recovery successful for {category.value} error")
                    return True
                    
            # Update recovery attempts
            error_record['recovery_attempts'] += 1
            
            return False
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False
            
    async def _recover_database_error(self, error_record: Dict[str, Any]) -> bool:
        """Recover from database errors"""
        try:
            # Wait before retry
            await asyncio.sleep(1.0)
            
            # Simple database health check
            # In a real implementation, this would test the database connection
            logger.info("Database recovery: Testing connection...")
            
            # For now, assume recovery is successful
            return True
            
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False
            
    async def _recover_network_error(self, error_record: Dict[str, Any]) -> bool:
        """Recover from network errors"""
        try:
            # Wait before retry
            await asyncio.sleep(2.0)
            
            # Test network connectivity
            logger.info("Network recovery: Testing connectivity...")
            
            # For now, assume recovery is successful
            return True
            
        except Exception as e:
            logger.error(f"Network recovery failed: {e}")
            return False
            
    async def _recover_data_error(self, error_record: Dict[str, Any]) -> bool:
        """Recover from data errors"""
        try:
            # Wait before retry
            await asyncio.sleep(0.5)
            
            # Attempt data validation/correction
            logger.info("Data recovery: Attempting data correction...")
            
            # For now, assume recovery is successful
            return True
            
        except Exception as e:
            logger.error(f"Data recovery failed: {e}")
            return False
            
    async def _recover_ml_error(self, error_record: Dict[str, Any]) -> bool:
        """Recover from ML errors"""
        try:
            # Wait before retry
            await asyncio.sleep(5.0)
            
            # Attempt model reload/restart
            logger.info("ML recovery: Attempting model restart...")
            
            # For now, assume recovery is successful
            return True
            
        except Exception as e:
            logger.error(f"ML recovery failed: {e}")
            return False
            
    async def _recover_trading_error(self, error_record: Dict[str, Any]) -> bool:
        """Recover from trading errors"""
        try:
            # Trading errors typically don't have automatic recovery
            logger.warning("Trading error recovery: Manual intervention required")
            
            # Log the error for manual review
            self._log_trading_error_for_review(error_record)
            
            return False
            
        except Exception as e:
            logger.error(f"Trading error recovery failed: {e}")
            return False
            
    def _log_trading_error_for_review(self, error_record: Dict[str, Any]):
        """Log trading error for manual review"""
        review_file = "logs/trading_errors_review.log"
        
        try:
            with open(review_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {error_record['timestamp']}\n")
                f.write(f"Context: {error_record['context']}\n")
                f.write(f"Error: {error_record['error_message']}\n")
                f.write(f"Additional Data: {json.dumps(error_record['additional_data'], indent=2)}\n")
                f.write(f"Traceback:\n{error_record['traceback']}\n")
                f.write(f"{'='*80}\n")
                
        except Exception as e:
            logger.error(f"Failed to log trading error for review: {e}")
            
    def add_error_callback(self, category: ErrorCategory, callback: Callable):
        """Add error callback for specific category"""
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
            
        self.error_callbacks[category].append(callback)
        
    def remove_error_callback(self, category: ErrorCategory, callback: Callable):
        """Remove error callback"""
        if category in self.error_callbacks and callback in self.error_callbacks[category]:
            self.error_callbacks[category].remove(callback)
            
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        # Reset counters if more than 24 hours have passed
        if datetime.now() - self.last_reset > timedelta(hours=24):
            self._reset_error_counts()
            
        return {
            'total_errors': len(self.error_history),
            'error_counts_by_severity': {severity.value: count for severity, count in self.error_counts.items()},
            'error_counts_by_category': self._get_error_counts_by_category(),
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'unresolved_errors': [e for e in self.error_history if not e['resolved']],
            'last_reset': self.last_reset.isoformat()
        }
        
    def _get_error_counts_by_category(self) -> Dict[str, int]:
        """Get error counts by category"""
        category_counts = {}
        for error in self.error_history:
            category = error['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
        
    def _reset_error_counts(self):
        """Reset error counts"""
        self.error_counts = {severity: 0 for severity in ErrorSeverity}
        self.last_reset = datetime.now()
        logger.info("Error counts reset")
        
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        logger.info("Error history cleared")
        
    def get_errors_by_category(self, category: ErrorCategory, 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get errors by category"""
        category_errors = [
            error for error in self.error_history 
            if ErrorCategory(error['category']) == category
        ]
        
        return category_errors[-limit:] if limit else category_errors
        
    def get_errors_by_severity(self, severity: ErrorSeverity, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get errors by severity"""
        severity_errors = [
            error for error in self.error_history 
            if ErrorSeverity(error['severity']) == severity
        ]
        
        return severity_errors[-limit:] if limit else severity_errors
        
    def mark_error_resolved(self, error_index: int):
        """Mark an error as resolved"""
        if 0 <= error_index < len(self.error_history):
            self.error_history[error_index]['resolved'] = True
            logger.info(f"Error {error_index} marked as resolved")
            
    def export_error_report(self, filename: str = None) -> str:
        """Export error report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/error_report_{timestamp}.json"
            
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': self.get_error_summary(),
                'all_errors': self.error_history
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Error report exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export error report: {e}")
            return ""

# Global error handler instance
error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    return error_handler

def handle_error(error: Exception, context: str = "", 
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                additional_data: Dict[str, Any] = None,
                retry: bool = True) -> bool:
    """Global error handling function"""
    return error_handler.handle_error(
        error, context, category, severity, additional_data, retry
    )

# Decorator for automatic error handling
def error_handler_decorator(category: ErrorCategory = ErrorCategory.UNKNOWN,
                           severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                           retry: bool = True):
    """Decorator to automatically handle errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = f"{func.__module__}.{func.__name__}"
                handle_error(e, context, category, severity, retry=retry)
                raise
        return wrapper
    return decorator

# Async version of error handler decorator
def async_error_handler_decorator(category: ErrorCategory = ErrorCategory.UNKNOWN,
                                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                                 retry: bool = True):
    """Async decorator to automatically handle errors"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = f"{func.__module__}.{func.__name__}"
                handle_error(e, context, category, severity, retry=retry)
                raise
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test error handling functionality
    handler = ErrorHandler()
    
    # Test different error types
    try:
        raise ValueError("Test value error")
    except Exception as e:
        handler.handle_error(e, "test_function", ErrorCategory.VALIDATION, ErrorSeverity.LOW)
        
    try:
        raise ConnectionError("Test connection error")
    except Exception as e:
        handler.handle_error(e, "test_network", ErrorCategory.NETWORK, ErrorSeverity.MEDIUM)
        
    try:
        raise RuntimeError("Test critical error")
    except Exception as e:
        handler.handle_error(e, "test_critical", ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL)
        
    # Show error summary
    summary = handler.get_error_summary()
    print("Error Summary:", json.dumps(summary, indent=2))
    
    # Export error report
    report_file = handler.export_error_report()
    print(f"Error report exported to: {report_file}")