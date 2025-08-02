# data_sanitizer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
import re
from scipy import stats

class DataSanitizer:
    """
    A class to perform preliminary checks and basic cleaning tasks on raw incoming financial data.
    Removes obviously corrupted records, handles missing values, and standardizes formats.
    """
    
    def __init__(self):
        """Initialize Data Sanitizer"""
        self.logger = self._setup_logger()
        
        # Sanitization rules
        self.sanitization_rules = {}
        self.field_types = {}
        self.field_ranges = {}
        self.missing_value_handlers = {}
        
        # Sanitization status
        self.is_sanitizing = False
        self.sanitization_thread = None
        
        # Sanitization metrics
        self.sanitization_metrics = {
            'total_processed': 0,
            'total_cleaned': 0,
            'total_removed': 0,
            'total_missing': 0,
            'total_outliers': 0,
            'total_duplicates': 0,
            'last_sanitization_time': None
        }
        
        # Data streams
        self.data_streams = {}
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("DataSanitizer")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("data_sanitizer.log")
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
    
    def add_data_stream(self, stream_id, stream_name=None):
        """
        Add data stream for sanitization
        
        Args:
            stream_id (str): Stream identifier
            stream_name (str): Stream display name
            
        Returns:
            bool: Whether successfully added stream
        """
        if stream_id in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} already exists")
            return False
        
        # Add stream
        self.data_streams[stream_id] = {
            'stream_name': stream_name or stream_id,
            'created_at': datetime.now()
        }
        
        # Initialize sanitization rules
        self.sanitization_rules[stream_id] = {
            'remove_duplicates': True,
            'remove_outliers': True,
            'handle_missing_values': True,
            'outlier_method': 'z_score',  # z_score, iqr, isolation_forest
            'outlier_threshold': 3.0,  # For z_score
            'iqr_factor': 1.5,  # For IQR
            'contamination': 0.1  # For isolation_forest
        }
        
        # Initialize field types
        self.field_types[stream_id] = {}
        
        # Initialize field ranges
        self.field_ranges[stream_id] = {}
        
        # Initialize missing value handlers
        self.missing_value_handlers[stream_id] = {}
        
        self.logger.info(f"Added data stream {stream_id} for sanitization")
        return True
    
    def remove_data_stream(self, stream_id):
        """
        Remove data stream from sanitization
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            bool: Whether successfully removed stream
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Remove stream and related configurations
        del self.data_streams[stream_id]
        del self.sanitization_rules[stream_id]
        del self.field_types[stream_id]
        del self.field_ranges[stream_id]
        del self.missing_value_handlers[stream_id]
        
        self.logger.info(f"Removed data stream {stream_id} from sanitization")
        return True
    
    def set_sanitization_rules(self, stream_id, rules):
        """
        Set sanitization rules for a stream
        
        Args:
            stream_id (str): Stream identifier
            rules (dict): Sanitization rules
            
        Returns:
            bool: Whether successfully set rules
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Validate rules
        valid_rules = [
            'remove_duplicates', 'remove_outliers', 'handle_missing_values',
            'outlier_method', 'outlier_threshold', 'iqr_factor', 'contamination'
        ]
        
        for rule in rules:
            if rule not in valid_rules:
                self.logger.error(f"Invalid sanitization rule: {rule}")
                return False
        
        # Set rules
        self.sanitization_rules[stream_id].update(rules)
        
        self.logger.info(f"Set sanitization rules for {stream_id}")
        return True
    
    def set_field_types(self, stream_id, field_types):
        """
        Set expected field types for a stream
        
        Args:
            stream_id (str): Stream identifier
            field_types (dict): Field types (field_name -> type)
            
        Returns:
            bool: Whether successfully set field types
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Validate field types
        valid_types = ['numeric', 'string', 'datetime', 'boolean']
        for field, field_type in field_types.items():
            if field_type not in valid_types:
                self.logger.error(f"Invalid field type for {field}: {field_type}")
                return False
        
        # Set field types
        self.field_types[stream_id] = field_types.copy()
        
        self.logger.info(f"Set field types for {stream_id}")
        return True
    
    def set_field_ranges(self, stream_id, field_ranges):
        """
        Set expected field ranges for a stream
        
        Args:
            stream_id (str): Stream identifier
            field_ranges (dict): Field ranges (field_name -> (min, max))
            
        Returns:
            bool: Whether successfully set field ranges
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Validate field ranges
        for field, (min_val, max_val) in field_ranges.items():
            if min_val >= max_val:
                self.logger.error(f"Invalid range for {field}: min ({min_val}) must be less than max ({max_val})")
                return False
        
        # Set field ranges
        self.field_ranges[stream_id] = field_ranges.copy()
        
        self.logger.info(f"Set field ranges for {stream_id}")
        return True
    
    def set_missing_value_handler(self, stream_id, field, handler):
        """
        Set missing value handler for a field
        
        Args:
            stream_id (str): Stream identifier
            field (str): Field name
            handler (dict): Handler configuration
            
        Returns:
            bool: Whether successfully set handler
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Validate handler
        valid_methods = ['drop', 'fill', 'interpolate', 'forward_fill', 'backward_fill']
        method = handler.get('method')
        
        if method not in valid_methods:
            self.logger.error(f"Invalid missing value method for {field}: {method}")
            return False
        
        if method == 'fill' and 'value' not in handler:
            self.logger.error(f"Missing value for fill method for {field}")
            return False
        
        # Set handler
        if stream_id not in self.missing_value_handlers:
            self.missing_value_handlers[stream_id] = {}
        
        self.missing_value_handlers[stream_id][field] = handler
        
        self.logger.info(f"Set missing value handler for {stream_id}.{field}")
        return True
    
    def sanitize_data(self, stream_id, data):
        """
        Sanitize data
        
        Args:
            stream_id (str): Stream identifier
            data (dict or list): Data to sanitize
            
        Returns:
            dict: Sanitization result
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return None
        
        # Convert to list if single record
        if isinstance(data, dict):
            data = [data]
        
        if not data:
            return {
                'sanitized_data': [],
                'removed_count': 0,
                'cleaned_count': 0,
                'missing_count': 0,
                'outlier_count': 0,
                'duplicate_count': 0,
                'issues': []
            }
        
        # Get sanitization rules
        rules = self.sanitization_rules[stream_id]
        
        # Initialize result
        result = {
            'sanitized_data': data.copy(),
            'removed_count': 0,
            'cleaned_count': 0,
            'missing_count': 0,
            'outlier_count': 0,
            'duplicate_count': 0,
            'issues': []
        }
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)
        
        # Record original count
        original_count = len(df)
        
        # Handle missing values
        if rules.get('handle_missing_values', True):
            missing_result = self._handle_missing_values(stream_id, df)
            result['sanitized_data'] = missing_result['data']
            result['missing_count'] = missing_result['missing_count']
            result['issues'].extend(missing_result['issues'])
            
            if missing_result['cleaned_count'] > 0:
                result['cleaned_count'] += missing_result['cleaned_count']
        
        # Remove duplicates
        if rules.get('remove_duplicates', True):
            duplicate_result = self._remove_duplicates(df)
            result['sanitized_data'] = duplicate_result['data']
            result['duplicate_count'] = duplicate_result['duplicate_count']
            
            if duplicate_result['duplicate_count'] > 0:
                result['issues'].append(f"Removed {duplicate_result['duplicate_count']} duplicate records")
        
        # Remove outliers
        if rules.get('remove_outliers', True):
            outlier_result = self._remove_outliers(stream_id, df)
            result['sanitized_data'] = outlier_result['data']
            result['outlier_count'] = outlier_result['outlier_count']
            
            if outlier_result['outlier_count'] > 0:
                result['issues'].append(f"Removed {outlier_result['outlier_count']} outlier records")
        
        # Convert back to list of dictionaries
        result['sanitized_data'] = result['sanitized_data'].to_dict('records')
        
        # Calculate removed count
        result['removed_count'] = original_count - len(result['sanitized_data'])
        
        # Update metrics
        self.sanitization_metrics['total_processed'] += original_count
        self.sanitization_metrics['total_cleaned'] += result['cleaned_count']
        self.sanitization_metrics['total_removed'] += result['removed_count']
        self.sanitization_metrics['total_missing'] += result['missing_count']
        self.sanitization_metrics['total_outliers'] += result['outlier_count']
        self.sanitization_metrics['total_duplicates'] += result['duplicate_count']
        self.sanitization_metrics['last_sanitization_time'] = datetime.now()
        
        return result
    
    def _handle_missing_values(self, stream_id, df):
        """
        Handle missing values in DataFrame
        
        Args:
            stream_id (str): Stream identifier
            df (pandas.DataFrame): DataFrame to process
            
        Returns:
            dict: Processing result
        """
        result = {
            'data': df.copy(),
            'missing_count': 0,
            'cleaned_count': 0,
            'issues': []
        }
        
        # Get missing value handlers
        handlers = self.missing_value_handlers.get(stream_id, {})
        
        # Check each column for missing values
        for column in df.columns:
            # Count missing values
            missing_count = df[column].isna().sum()
            
            if missing_count == 0:
                continue
            
            result['missing_count'] += missing_count
            
            # Get handler for column
            handler = handlers.get(column, {'method': 'drop'})
            method = handler.get('method', 'drop')
            
            # Handle missing values based on method
            if method == 'drop':
                # Drop rows with missing values
                result['data'] = result['data'].dropna(subset=[column])
                result['issues'].append(f"Dropped {missing_count} rows with missing values in {column}")
            elif method == 'fill':
                # Fill with specified value
                fill_value = handler.get('value')
                result['data'][column] = result['data'][column].fillna(fill_value)
                result['issues'].append(f"Filled {missing_count} missing values in {column} with {fill_value}")
            elif method == 'interpolate':
                # Interpolate missing values
                if pd.api.types.is_numeric_dtype(result['data'][column]):
                    result['data'][column] = result['data'][column].interpolate()
                    result['issues'].append(f"Interpolated {missing_count} missing values in {column}")
                else:
                    # Can't interpolate non-numeric data, drop instead
                    result['data'] = result['data'].dropna(subset=[column])
                    result['issues'].append(f"Dropped {missing_count} rows with missing non-numeric values in {column}")
            elif method == 'forward_fill':
                # Forward fill missing values
                result['data'][column] = result['data'][column].fillna(method='ffill')
                result['issues'].append(f"Forward filled {missing_count} missing values in {column}")
            elif method == 'backward_fill':
                # Backward fill missing values
                result['data'][column] = result['data'][column].fillna(method='bfill')
                result['issues'].append(f"Backward filled {missing_count} missing values in {column}")
            
            result['cleaned_count'] += missing_count
        
        return result
    
    def _remove_duplicates(self, df):
        """
        Remove duplicate records from DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame to process
            
        Returns:
            dict: Processing result
        """
        result = {
            'data': df.copy(),
            'duplicate_count': 0
        }
        
        # Count duplicates
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count > 0:
            # Remove duplicates
            result['data'] = result['data'].drop_duplicates()
            result['duplicate_count'] = duplicate_count
        
        return result
    
    def _remove_outliers(self, stream_id, df):
        """
        Remove outlier records from DataFrame
        
        Args:
            stream_id (str): Stream identifier
            df (pandas.DataFrame): DataFrame to process
            
        Returns:
            dict: Processing result
        """
        result = {
            'data': df.copy(),
            'outlier_count': 0
        }
        
        # Get sanitization rules
        rules = self.sanitization_rules[stream_id]
        method = rules.get('outlier_method', 'z_score')
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return result
        
        # Remove outliers based on method
        if method == 'z_score':
            threshold = rules.get('outlier_threshold', 3.0)
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(df[numeric_columns]))
            
            # Find outliers
            outliers = (z_scores > threshold).any(axis=1)
            
            # Remove outliers
            result['data'] = result['data'][~outliers]
            result['outlier_count'] = outliers.sum()
            
        elif method == 'iqr':
            factor = rules.get('iqr_factor', 1.5)
            
            # Calculate IQR for each column
            q1 = df[numeric_columns].quantile(0.25)
            q3 = df[numeric_columns].quantile(0.75)
            iqr = q3 - q1
            
            # Define outlier bounds
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            
            # Find outliers
            outliers = ((df[numeric_columns] < lower_bound) | (df[numeric_columns] > upper_bound)).any(axis=1)
            
            # Remove outliers
            result['data'] = result['data'][~outliers]
            result['outlier_count'] = outliers.sum()
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            contamination = rules.get('contamination', 0.1)
            
            # Fit isolation forest
            clf = IsolationForest(contamination=contamination, random_state=42)
            outliers = clf.fit_predict(df[numeric_columns]) == -1
            
            # Remove outliers
            result['data'] = result['data'][~outliers]
            result['outlier_count'] = outliers.sum()
        
        return result
    
    def get_sanitization_metrics(self):
        """
        Get sanitization metrics
        
        Returns:
            dict: Sanitization metrics
        """
        return self.sanitization_metrics.copy()
    
    def get_sanitization_rules(self, stream_id):
        """
        Get sanitization rules for a stream
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            dict: Sanitization rules or None if stream not found
        """
        if stream_id not in self.sanitization_rules:
            self.logger.warning(f"Sanitization rules for {stream_id} not found")
            return None
        
        return self.sanitization_rules[stream_id].copy()
    
    def get_field_types(self, stream_id):
        """
        Get field types for a stream
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            dict: Field types or None if stream not found
        """
        if stream_id not in self.field_types:
            self.logger.warning(f"Field types for {stream_id} not found")
            return None
        
        return self.field_types[stream_id].copy()
    
    def get_field_ranges(self, stream_id):
        """
        Get field ranges for a stream
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            dict: Field ranges or None if stream not found
        """
        if stream_id not in self.field_ranges:
            self.logger.warning(f"Field ranges for {stream_id} not found")
            return None
        
        return self.field_ranges[stream_id].copy()
    
    def get_missing_value_handlers(self, stream_id):
        """
        Get missing value handlers for a stream
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            dict: Missing value handlers or None if stream not found
        """
        if stream_id not in self.missing_value_handlers:
            self.logger.warning(f"Missing value handlers for {stream_id} not found")
            return None
        
        return self.missing_value_handlers[stream_id].copy()
    
    def reset_sanitization_metrics(self):
        """
        Reset sanitization metrics
        
        Returns:
            bool: Whether successfully reset metrics
        """
        self.sanitization_metrics = {
            'total_processed': 0,
            'total_cleaned': 0,
            'total_removed': 0,
            'total_missing': 0,
            'total_outliers': 0,
            'total_duplicates': 0,
            'last_sanitization_time': None
        }
        
        self.logger.info("Reset sanitization metrics")
        return True
