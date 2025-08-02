# feed_integrity_logger.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
import os
import json

class FeedIntegrityLogger:
    """
    A class to maintain detailed logs of all validation checks, issues, and data quality warnings.
    Provides audit trail for data quality analysis and troubleshooting.
    """
    
    def __init__(self, log_dir="logs/feed_integrity"):
        """
        Initialize Feed Integrity Logger
        
        Args:
            log_dir (str): Directory to store log files
        """
        self.log_dir = log_dir
        self.logger = self._setup_logger()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Integrity logs
        self.integrity_logs = {}
        self.log_files = {}
        
        # Logging status
        self.is_logging = False
        self.logging_thread = None
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("FeedIntegrityLogger")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "feed_integrity_logger.log"))
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def add_feed(self, feed_id, feed_name=None):
        """
        Add feed for integrity logging
        
        Args:
            feed_id (str): Feed identifier
            feed_name (str): Feed display name
            
        Returns:
            bool: Whether successfully added feed
        """
        if feed_id in self.integrity_logs:
            self.logger.warning(f"Feed {feed_id} is already being logged")
            return False
        
        # Create log file for feed
        log_file = os.path.join(self.log_dir, f"{feed_id}_integrity.log")
        self.log_files[feed_id] = log_file
        
        # Initialize integrity logs
        self.integrity_logs[feed_id] = {
            'feed_name': feed_name or feed_id,
            'created_at': datetime.now(),
            'events': [],
            'stats': {
                'total_events': 0,
                'error_count': 0,
                'warning_count': 0,
                'info_count': 0
            }
        }
        
        # If this is the first feed, start the logging thread
        if not self.is_logging:
            self.is_logging = True
            self.logging_thread = threading.Thread(target=self._log_integrity)
            self.logging_thread.daemon = True
            self.logging_thread.start()
        
        self.logger.info(f"Added feed {feed_id} for integrity logging")
        return True
    
    def remove_feed(self, feed_id):
        """
        Remove feed from integrity logging
        
        Args:
            feed_id (str): Feed identifier
            
        Returns:
            bool: Whether successfully removed feed
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Feed {feed_id} is not being logged")
            return False
        
        # Save final logs before removing
        self._save_logs(feed_id)
        
        # Remove from logs and files
        del self.integrity_logs[feed_id]
        del self.log_files[feed_id]
        
        # If no more feeds, stop the logging thread
        if not self.integrity_logs and self.is_logging:
            self.is_logging = False
            if self.logging_thread and self.logging_thread.is_alive():
                self.logging_thread.join(timeout=5)
        
        self.logger.info(f"Removed feed {feed_id} from integrity logging")
        return True
    
    def log_event(self, feed_id, event_type, message, details=None, severity='info'):
        """
        Log integrity event
        
        Args:
            feed_id (str): Feed identifier
            event_type (str): Type of event
            message (str): Event message
            details (dict): Additional event details
            severity (str): Event severity ('error', 'warning', 'info')
            
        Returns:
            bool: Whether successfully logged event
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Feed {feed_id} is not being logged")
            return False
        
        # Validate severity
        if severity not in ['error', 'warning', 'info']:
            self.logger.error(f"Invalid severity: {severity}")
            return False
        
        # Create event
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'message': message,
            'details': details or {},
            'severity': severity
        }
        
        # Add to logs
        self.integrity_logs[feed_id]['events'].append(event)
        
        # Update stats
        self.integrity_logs[feed_id]['stats']['total_events'] += 1
        if severity == 'error':
            self.integrity_logs[feed_id]['stats']['error_count'] += 1
        elif severity == 'warning':
            self.integrity_logs[feed_id]['stats']['warning_count'] += 1
        else:
            self.integrity_logs[feed_id]['stats']['info_count'] += 1
        
        # Log to file immediately for critical events
        if severity == 'error':
            self._log_to_file(feed_id, event)
        
        return True
    
    def log_validation_result(self, feed_id, validation_result):
        """
        Log validation result
        
        Args:
            feed_id (str): Feed identifier
            validation_result (dict): Validation result
            
        Returns:
            bool: Whether successfully logged result
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Feed {feed_id} is not being logged")
            return False
        
        # Determine severity based on validation result
        if not validation_result.get('valid', True):
            severity = 'error'
        elif validation_result.get('warnings'):
            severity = 'warning'
        else:
            severity = 'info'
        
        # Log event
        return self.log_event(
            feed_id=feed_id,
            event_type='validation',
            message=f"Data validation {'passed' if severity == 'info' else 'failed'}",
            details=validation_result,
            severity=severity
        )
    
    def log_anomaly(self, feed_id, anomaly):
        """
        Log anomaly
        
        Args:
            feed_id (str): Feed identifier
            anomaly (dict): Anomaly details
            
        Returns:
            bool: Whether successfully logged anomaly
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Feed {feed_id} is not being logged")
            return False
        
        # Determine severity based on anomaly
        severity = anomaly.get('severity', 'medium')
        if severity == 'high':
            log_severity = 'error'
        elif severity == 'medium':
            log_severity = 'warning'
        else:
            log_severity = 'info'
        
        # Log event
        return self.log_event(
            feed_id=feed_id,
            event_type='anomaly',
            message=anomaly.get('description', 'Anomaly detected'),
            details=anomaly,
            severity=log_severity
        )
    
    def log_data_gap(self, feed_id, gap_start, gap_end, expected_count, actual_count):
        """
        Log data gap
        
        Args:
            feed_id (str): Feed identifier
            gap_start (datetime): Gap start time
            gap_end (datetime): Gap end time
            expected_count (int): Expected data point count
            actual_count (int): Actual data point count
            
        Returns:
            bool: Whether successfully logged gap
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Feed {feed_id} is not being logged")
            return False
        
        # Calculate gap duration
        gap_duration = (gap_end - gap_start).total_seconds()
        
        # Calculate gap percentage
        gap_percentage = (expected_count - actual_count) / expected_count * 100 if expected_count > 0 else 100
        
        # Determine severity based on gap percentage
        if gap_percentage > 50:
            severity = 'error'
        elif gap_percentage > 10:
            severity = 'warning'
        else:
            severity = 'info'
        
        # Log event
        return self.log_event(
            feed_id=feed_id,
            event_type='data_gap',
            message=f"Data gap detected: {gap_percentage:.1f}% missing data",
            details={
                'gap_start': gap_start,
                'gap_end': gap_end,
                'gap_duration_seconds': gap_duration,
                'expected_count': expected_count,
                'actual_count': actual_count,
                'missing_count': expected_count - actual_count,
                'gap_percentage': gap_percentage
            },
            severity=severity
        )
    
    def log_latency(self, feed_id, latency_ms, threshold_ms=1000):
        """
        Log data latency
        
        Args:
            feed_id (str): Feed identifier
            latency_ms (float): Latency in milliseconds
            threshold_ms (float): Threshold for high latency
            
        Returns:
            bool: Whether successfully logged latency
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Feed {feed_id} is not being logged")
            return False
        
        # Determine severity based on latency
        if latency_ms > threshold_ms * 3:
            severity = 'error'
        elif latency_ms > threshold_ms:
            severity = 'warning'
        else:
            severity = 'info'
        
        # Log event
        return self.log_event(
            feed_id=feed_id,
            event_type='latency',
            message=f"Data latency: {latency_ms:.2f}ms",
            details={
                'latency_ms': latency_ms,
                'threshold_ms': threshold_ms
            },
            severity=severity
        )
    
    def _log_integrity(self):
        """
        Internal method to log integrity events, runs in separate thread
        """
        while self.is_logging:
            try:
                # Process each feed
                for feed_id in self.integrity_logs:
                    # Save logs to file
                    self._save_logs(feed_id)
                
                # Sleep for a while
                time.sleep(60)  # Save logs every minute
                
            except Exception as e:
                self.logger.error(f"Error logging integrity: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _save_logs(self, feed_id):
        """
        Save logs to file
        
        Args:
            feed_id (str): Feed identifier
        """
        if feed_id not in self.integrity_logs:
            return
        
        try:
            # Get log file path
            log_file = self.log_files[feed_id]
            
            # Get logs
            logs = self.integrity_logs[feed_id]
            
            # Prepare log data
            log_data = {
                'feed_id': feed_id,
                'feed_name': logs['feed_name'],
                'created_at': logs['created_at'].isoformat(),
                'stats': logs['stats'],
                'events': []
            }
            
            # Add events (convert datetime objects to strings)
            for event in logs['events']:
                event_copy = event.copy()
                event_copy['timestamp'] = event['timestamp'].isoformat()
                log_data['events'].append(event_copy)
            
            # Write to file
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            self.logger.debug(f"Saved integrity logs for {feed_id} to {log_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving logs for {feed_id}: {str(e)}")
    
    def _log_to_file(self, feed_id, event):
        """
        Log event to file immediately
        
        Args:
            feed_id (str): Feed identifier
            event (dict): Event to log
        """
        if feed_id not in self.log_files:
            return
        
        try:
            # Get log file path
            log_file = self.log_files[feed_id]
            
            # Prepare log entry
            log_entry = {
                'timestamp': event['timestamp'].isoformat(),
                'type': event['type'],
                'message': event['message'],
                'details': event['details'],
                'severity': event['severity']
            }
            
            # Append to file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
        except Exception as e:
            self.logger.error(f"Error logging event to file for {feed_id}: {str(e)}")
    
    def get_integrity_logs(self, feed_id, count=None):
        """
        Get integrity logs for a feed
        
        Args:
            feed_id (str): Feed identifier
            count (int): Number of events to get, None means get all
            
        Returns:
            dict: Integrity logs or None if feed not found
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Integrity logs for {feed_id} not found")
            return None
        
        logs = self.integrity_logs[feed_id].copy()
        
        # Limit events if requested
        if count is not None:
            logs['events'] = logs['events'][-count:]
        
        return logs
    
    def get_integrity_logs_dataframe(self, feed_id, count=None):
        """
        Get integrity logs for a feed as DataFrame
        
        Args:
            feed_id (str): Feed identifier
            count (int): Number of events to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing integrity logs or None if feed not found
        """
        logs = self.get_integrity_logs(feed_id, count)
        
        if logs is None:
            return None
        
        if not logs['events']:
            return pd.DataFrame()
        
        # Create DataFrame from events
        df = pd.DataFrame(logs['events'])
        
        # Convert timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add feed information
        df['feed_id'] = feed_id
        df['feed_name'] = logs['feed_name']
        
        return df
    
    def get_feed_stats(self, feed_id):
        """
        Get feed statistics
        
        Args:
            feed_id (str): Feed identifier
            
        Returns:
            dict: Feed statistics or None if feed not found
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Integrity logs for {feed_id} not found")
            return None
        
        return self.integrity_logs[feed_id]['stats'].copy()
    
    def save_logs_to_csv(self, feed_id, filename, count=None):
        """
        Save integrity logs for a feed to CSV file
        
        Args:
            feed_id (str): Feed identifier
            filename (str): Filename
            count (int): Number of events to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_integrity_logs_dataframe(feed_id, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No integrity logs to save for feed {feed_id}")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} integrity log records for {feed_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save integrity logs for {feed_id} to CSV: {str(e)}")
            return False
    
    def save_logs_to_parquet(self, feed_id, filename, count=None):
        """
        Save integrity logs for a feed to Parquet file
        
        Args:
            feed_id (str): Feed identifier
            filename (str): Filename
            count (int): Number of events to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_integrity_logs_dataframe(feed_id, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No integrity logs to save for feed {feed_id}")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} integrity log records for {feed_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save integrity logs for {feed_id} to Parquet: {str(e)}")
            return False
    
    def clear_logs(self, feed_id):
        """
        Clear logs for a feed
        
        Args:
            feed_id (str): Feed identifier
            
        Returns:
            bool: Whether successfully cleared logs
        """
        if feed_id not in self.integrity_logs:
            self.logger.warning(f"Integrity logs for {feed_id} not found")
            return False
        
        # Clear events and reset stats
        self.integrity_logs[feed_id]['events'] = []
        self.integrity_logs[feed_id]['stats'] = {
            'total_events': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0
        }
        
        self.logger.info(f"Cleared integrity logs for {feed_id}")
        return True
