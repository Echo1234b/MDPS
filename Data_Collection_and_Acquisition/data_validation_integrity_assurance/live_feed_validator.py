# live_feed_validator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
import re

class LiveFeedValidator:
    """
    A class to validate data structure, format, and expected fields for live market data feeds.
    Implements real-time checks to ensure data quality and consistency.
    """
    
    def __init__(self):
        """Initialize Live Feed Validator"""
        self.logger = self._setup_logger()
        
        # Validation rules
        self.validation_rules = {}
        self.field_types = {}
        self.field_ranges = {}
        self.required_fields = {}
        
        # Validation status
        self.is_validating = False
        self.validation_thread = None
        
        # Validation results
        self.validation_results = {}
        self.validation_history = {}
        
        # Callbacks
        self.validation_callbacks = {}
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("LiveFeedValidator")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("live_feed_validator.log")
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
    
    def add_validation_rules(self, feed_id, rules):
        """
        Add validation rules for a feed
        
        Args:
            feed_id (str): Feed identifier
            rules (dict): Validation rules
            
        Returns:
            bool: Whether successfully added rules
        """
        if feed_id in self.validation_rules:
            self.logger.warning(f"Validation rules for {feed_id} already exist")
            return False
        
        # Validate rules format
        if not isinstance(rules, dict):
            self.logger.error("Rules must be a dictionary")
            return False
        
        # Add rules
        self.validation_rules[feed_id] = rules
        
        # Initialize validation results and history
        self.validation_results[feed_id] = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'last_check': None
        }
        
        self.validation_history[feed_id] = []
        
        self.logger.info(f"Added validation rules for feed {feed_id}")
        return True
    
    def remove_validation_rules(self, feed_id):
        """
        Remove validation rules for a feed
        
        Args:
            feed_id (str): Feed identifier
            
        Returns:
            bool: Whether successfully removed rules
        """
        if feed_id not in self.validation_rules:
            self.logger.warning(f"Validation rules for {feed_id} not found")
            return False
        
        # Remove rules
        del self.validation_rules[feed_id]
        del self.validation_results[feed_id]
        del self.validation_history[feed_id]
        
        # Remove callback if exists
        if feed_id in self.validation_callbacks:
            del self.validation_callbacks[feed_id]
        
        self.logger.info(f"Removed validation rules for feed {feed_id}")
        return True
    
    def set_field_types(self, feed_id, field_types):
        """
        Set expected field types for a feed
        
        Args:
            feed_id (str): Feed identifier
            field_types (dict): Field types (field_name -> type)
            
        Returns:
            bool: Whether successfully set field types
        """
        if feed_id not in self.validation_rules:
            self.logger.warning(f"Validation rules for {feed_id} not found")
            return False
        
        # Validate field types format
        if not isinstance(field_types, dict):
            self.logger.error("Field types must be a dictionary")
            return False
        
        # Set field types
        self.field_types[feed_id] = field_types
        
        self.logger.info(f"Set field types for feed {feed_id}")
        return True
    
    def set_field_ranges(self, feed_id, field_ranges):
        """
        Set expected field ranges for a feed
        
        Args:
            feed_id (str): Feed identifier
            field_ranges (dict): Field ranges (field_name -> (min, max))
            
        Returns:
            bool: Whether successfully set field ranges
        """
        if feed_id not in self.validation_rules:
            self.logger.warning(f"Validation rules for {feed_id} not found")
            return False
        
        # Validate field ranges format
        if not isinstance(field_ranges, dict):
            self.logger.error("Field ranges must be a dictionary")
            return False
        
        # Set field ranges
        self.field_ranges[feed_id] = field_ranges
        
        self.logger.info(f"Set field ranges for feed {feed_id}")
        return True
    
    def set_required_fields(self, feed_id, required_fields):
        """
        Set required fields for a feed
        
        Args:
            feed_id (str): Feed identifier
            required_fields (list): List of required field names
            
        Returns:
            bool: Whether successfully set required fields
        """
        if feed_id not in self.validation_rules:
            self.logger.warning(f"Validation rules for {feed_id} not found")
            return False
        
        # Validate required fields format
        if not isinstance(required_fields, list):
            self.logger.error("Required fields must be a list")
            return False
        
        # Set required fields
        self.required_fields[feed_id] = required_fields
        
        self.logger.info(f"Set required fields for feed {feed_id}")
        return True
    
    def set_validation_callback(self, feed_id, callback):
        """
        Set validation callback for a feed
        
        Args:
            feed_id (str): Feed identifier
            callback (function): Callback function to handle validation results
            
        Returns:
            bool: Whether successfully set callback
        """
        if feed_id not in self.validation_rules:
            self.logger.warning(f"Validation rules for {feed_id} not found")
            return False
        
        self.validation_callbacks[feed_id] = callback
        self.logger.info(f"Set validation callback for feed {feed_id}")
        return True
    
    def start_validation(self):
        """
        Start validation
        
        Returns:
            bool: Whether successfully started validation
        """
        if self.is_validating:
            self.logger.warning("Validation is already running")
            return False
        
        if not self.validation_rules:
            self.logger.error("No validation rules defined")
            return False
        
        # Start validation thread
        self.is_validating = True
        self.validation_thread = threading.Thread(target=self._validate_feeds)
        self.validation_thread.daemon = True
        self.validation_thread.start()
        
        self.logger.info("Started live feed validation")
        return True
    
    def stop_validation(self):
        """
        Stop validation
        
        Returns:
            bool: Whether successfully stopped validation
        """
        if not self.is_validating:
            self.logger.warning("Validation is not running")
            return False
        
        # Stop validation thread
        self.is_validating = False
        if self.validation_thread and self.validation_thread.is_alive():
            self.validation_thread.join(timeout=5)
        
        self.logger.info("Stopped live feed validation")
        return True
    
    def _validate_feeds(self):
        """
        Internal method to validate feeds, runs in separate thread
        """
        while self.is_validating:
            try:
                # Process each feed
                for feed_id in self.validation_rules:
                    # Get validation rules for feed
                    rules = self.validation_rules[feed_id]
                    
                    # Get data source function from rules
                    if 'data_source' not in rules:
                        continue
                    
                    data_source = rules['data_source']
                    
                    # Get data from source
                    data = data_source()
                    
                    if data is None:
                        continue
                    
                    # Validate data
                    validation_result = self._validate_data(feed_id, data)
                    
                    # Update validation results
                    self.validation_results[feed_id] = validation_result
                    
                    # Add to history
                    self.validation_history[feed_id].append({
                        'time': datetime.now(),
                        'valid': validation_result['valid'],
                        'errors': validation_result['errors'],
                        'warnings': validation_result['warnings']
                    })
                    
                    # Limit history size
                    if len(self.validation_history[feed_id]) > 1000:
                        self.validation_history[feed_id].pop(0)
                    
                    # Call callback if provided
                    if feed_id in self.validation_callbacks:
                        try:
                            self.validation_callbacks[feed_id](validation_result)
                        except Exception as e:
                            self.logger.error(f"Error in validation callback for {feed_id}: {str(e)}")
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error validating feeds: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _validate_data(self, feed_id, data):
        """
        Validate data for a feed
        
        Args:
            feed_id (str): Feed identifier
            data (dict or list): Data to validate
            
        Returns:
            dict: Validation result
        """
        # Initialize validation result
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'last_check': datetime.now()
        }
        
        try:
            # Check if data is a list (multiple records) or dict (single record)
            if isinstance(data, list):
                # Validate each record
                for i, record in enumerate(data):
                    record_result = self._validate_record(feed_id, record)
                    if not record_result['valid']:
                        result['valid'] = False
                        result['errors'].extend([f"Record {i}: {error}" for error in record_result['errors']])
                    result['warnings'].extend([f"Record {i}: {warning}" for warning in record_result['warnings']])
            elif isinstance(data, dict):
                # Validate single record
                record_result = self._validate_record(feed_id, data)
                if not record_result['valid']:
                    result['valid'] = False
                    result['errors'].extend(record_result['errors'])
                result['warnings'].extend(record_result['warnings'])
            else:
                # Invalid data type
                result['valid'] = False
                result['errors'].append(f"Invalid data type: {type(data)}")
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_record(self, feed_id, record):
        """
        Validate a single record
        
        Args:
            feed_id (str): Feed identifier
            record (dict): Record to validate
            
        Returns:
            dict: Validation result
        """
        # Initialize validation result
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if record is a dictionary
            if not isinstance(record, dict):
                result['valid'] = False
                result['errors'].append(f"Record is not a dictionary: {type(record)}")
                return result
            
            # Check required fields
            if feed_id in self.required_fields:
                for field in self.required_fields[feed_id]:
                    if field not in record:
                        result['valid'] = False
                        result['errors'].append(f"Missing required field: {field}")
            
            # Check field types
            if feed_id in self.field_types:
                for field, expected_type in self.field_types[feed_id].items():
                    if field in record:
                        value = record[field]
                        
                        # Check type
                        if expected_type == 'numeric':
                            if not isinstance(value, (int, float)) and not (isinstance(value, str) and re.match(r'^-?\d+(\.\d+)?$', value)):
                                result['valid'] = False
                                result['errors'].append(f"Field {field} is not numeric: {value} ({type(value)})")
                        elif expected_type == 'string':
                            if not isinstance(value, str):
                                result['valid'] = False
                                result['errors'].append(f"Field {field} is not a string: {value} ({type(value)})")
                        elif expected_type == 'datetime':
                            if not isinstance(value, datetime) and not (isinstance(value, str) and re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$', value)):
                                result['valid'] = False
                                result['errors'].append(f"Field {field} is not a datetime: {value} ({type(value)})")
                        elif expected_type == 'boolean':
                            if not isinstance(value, bool) and not (isinstance(value, str) and value.lower() in ('true', 'false')):
                                result['valid'] = False
                                result['errors'].append(f"Field {field} is not a boolean: {value} ({type(value)})")
            
            # Check field ranges
            if feed_id in self.field_ranges:
                for field, (min_val, max_val) in self.field_ranges[feed_id].items():
                    if field in record:
                        value = record[field]
                        
                        # Convert to numeric if needed
                        if isinstance(value, str) and re.match(r'^-?\d+(\.\d+)?$', value):
                            value = float(value)
                        
                        # Check range
                        if isinstance(value, (int, float)):
                            if value < min_val or value > max_val:
                                result['warnings'].append(f"Field {field} is out of range: {value} (expected: {min_val} to {max_val})")
            
            # Apply custom validation rules
            if feed_id in self.validation_rules:
                rules = self.validation_rules[feed_id]
                
                # Apply custom validation functions
                if 'custom_validators' in rules:
                    for field, validator in rules['custom_validators'].items():
                        if field in record:
                            try:
                                validation_result = validator(record[field])
                                if isinstance(validation_result, bool):
                                    if not validation_result:
                                        result['valid'] = False
                                        result['errors'].append(f"Custom validation failed for field {field}")
                                elif isinstance(validation_result, str):
                                    result['warnings'].append(f"Custom validation warning for field {field}: {validation_result}")
                                elif isinstance(validation_result, dict) and 'valid' in validation_result:
                                    if not validation_result['valid']:
                                        result['valid'] = False
                                        if 'message' in validation_result:
                                            result['errors'].append(f"Custom validation failed for field {field}: {validation_result['message']}")
                                        else:
                                            result['errors'].append(f"Custom validation failed for field {field}")
                            except Exception as e:
                                result['warnings'].append(f"Error in custom validator for field {field}: {str(e)}")
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Record validation error: {str(e)}")
        
        return result
    
    def get_validation_result(self, feed_id):
        """
        Get validation result for a feed
        
        Args:
            feed_id (str): Feed identifier
            
        Returns:
            dict: Validation result or None if feed not found
        """
        if feed_id not in self.validation_results:
            self.logger.warning(f"Validation results for {feed_id} not found")
            return None
        
        return self.validation_results[feed_id].copy()
    
    def get_validation_history(self, feed_id, count=None):
        """
        Get validation history for a feed
        
        Args:
            feed_id (str): Feed identifier
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of validation records or None if feed not found
        """
        if feed_id not in self.validation_history:
            self.logger.warning(f"Validation history for {feed_id} not found")
            return None
        
        if count is None:
            return self.validation_history[feed_id].copy()
        else:
            return self.validation_history[feed_id][-count:]
    
    def get_validation_history_dataframe(self, feed_id, count=None):
        """
        Get validation history for a feed as DataFrame
        
        Args:
            feed_id (str): Feed identifier
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing validation history or None if feed not found
        """
        history = self.get_validation_history(feed_id, count)
        
        if history is None:
            return None
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    def save_validation_history_to_csv(self, feed_id, filename, count=None):
        """
        Save validation history for a feed to CSV file
        
        Args:
            feed_id (str): Feed identifier
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_validation_history_dataframe(feed_id, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No validation history to save for feed {feed_id}")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} validation records for {feed_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save validation history for {feed_id} to CSV: {str(e)}")
            return False
    
    def save_validation_history_to_parquet(self, feed_id, filename, count=None):
        """
        Save validation history for a feed to Parquet file
        
        Args:
            feed_id (str): Feed identifier
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_validation_history_dataframe(feed_id, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No validation history to save for feed {feed_id}")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} validation records for {feed_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save validation history for {feed_id} to Parquet: {str(e)}")
            return False
