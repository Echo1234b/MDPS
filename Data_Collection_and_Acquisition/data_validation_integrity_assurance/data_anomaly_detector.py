# data_anomaly_detector.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class DataAnomalyDetector:
    """
    A class to detect outliers and anomalies such as sudden spikes, frozen prices, or gaps.
    Implements statistical and machine learning methods for anomaly detection.
    """
    
    def __init__(self, window_size=100, contamination=0.1):
        """
        Initialize Data Anomaly Detector
        
        Args:
            window_size (int): Window size for statistical methods
            contamination (float): Expected proportion of anomalies for Isolation Forest
        """
        self.window_size = window_size
        self.contamination = contamination
        self.logger = self._setup_logger()
        
        # Detection status
        self.is_detecting = False
        self.detection_thread = None
        
        # Data streams
        self.data_streams = {}
        self.data_history = {}
        
        # Anomaly detection models
        self.models = {}
        
        # Detection results
        self.detection_results = {}
        self.anomaly_history = {}
        
        # Callbacks
        self.anomaly_callbacks = {}
        
        # Thresholds
        self.z_score_threshold = 3.0
        self.iqr_factor = 1.5
        self.price_change_threshold = 0.1  # 10%
        self.volume_change_threshold = 5.0  # 5x
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("DataAnomalyDetector")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("data_anomaly_detector.log")
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
    
    def add_data_stream(self, stream_id, data_source, fields=None):
        """
        Add data stream for anomaly detection
        
        Args:
            stream_id (str): Stream identifier
            data_source (function): Function to get data
            fields (list): List of fields to monitor, None means monitor all
            
        Returns:
            bool: Whether successfully added stream
        """
        if stream_id in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} already exists")
            return False
        
        # Add stream
        self.data_streams[stream_id] = {
            'data_source': data_source,
            'fields': fields
        }
        
        # Initialize data history
        self.data_history[stream_id] = []
        
        # Initialize detection results
        self.detection_results[stream_id] = {
            'anomalies': [],
            'last_check': None
        }
        
        # Initialize anomaly history
        self.anomaly_history[stream_id] = []
        
        # Initialize model
        self.models[stream_id] = None
        
        # If this is the first stream, start the detection thread
        if not self.is_detecting:
            self.is_detecting = True
            self.detection_thread = threading.Thread(target=self._detect_anomalies)
            self.detection_thread.daemon = True
            self.detection_thread.start()
        
        self.logger.info(f"Added data stream {stream_id} for anomaly detection")
        return True
    
    def remove_data_stream(self, stream_id):
        """
        Remove data stream from anomaly detection
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            bool: Whether successfully removed stream
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Remove stream
        del self.data_streams[stream_id]
        del self.data_history[stream_id]
        del self.detection_results[stream_id]
        del self.anomaly_history[stream_id]
        del self.models[stream_id]
        
        # Remove callback if exists
        if stream_id in self.anomaly_callbacks:
            del self.anomaly_callbacks[stream_id]
        
        # If no more streams, stop the detection thread
        if not self.data_streams and self.is_detecting:
            self.is_detecting = False
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=5)
        
        self.logger.info(f"Removed data stream {stream_id} from anomaly detection")
        return True
    
    def set_anomaly_callback(self, stream_id, callback):
        """
        Set anomaly callback for a stream
        
        Args:
            stream_id (str): Stream identifier
            callback (function): Callback function to handle anomalies
            
        Returns:
            bool: Whether successfully set callback
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        self.anomaly_callbacks[stream_id] = callback
        self.logger.info(f"Set anomaly callback for stream {stream_id}")
        return True
    
    def set_z_score_threshold(self, threshold):
        """
        Set Z-score threshold for anomaly detection
        
        Args:
            threshold (float): Z-score threshold
            
        Returns:
            bool: Whether successfully set threshold
        """
        if threshold <= 0:
            self.logger.error("Z-score threshold must be positive")
            return False
        
        self.z_score_threshold = threshold
        self.logger.info(f"Set Z-score threshold to {threshold}")
        return True
    
    def set_iqr_factor(self, factor):
        """
        Set IQR factor for anomaly detection
        
        Args:
            factor (float): IQR factor
            
        Returns:
            bool: Whether successfully set factor
        """
        if factor <= 0:
            self.logger.error("IQR factor must be positive")
            return False
        
        self.iqr_factor = factor
        self.logger.info(f"Set IQR factor to {factor}")
        return True
    
    def set_price_change_threshold(self, threshold):
        """
        Set price change threshold for anomaly detection
        
        Args:
            threshold (float): Price change threshold (as a decimal, e.g., 0.1 for 10%)
            
        Returns:
            bool: Whether successfully set threshold
        """
        if threshold <= 0:
            self.logger.error("Price change threshold must be positive")
            return False
        
        self.price_change_threshold = threshold
        self.logger.info(f"Set price change threshold to {threshold}")
        return True
    
    def set_volume_change_threshold(self, threshold):
        """
        Set volume change threshold for anomaly detection
        
        Args:
            threshold (float): Volume change threshold (as a multiplier, e.g., 5.0 for 5x)
            
        Returns:
            bool: Whether successfully set threshold
        """
        if threshold <= 1:
            self.logger.error("Volume change threshold must be greater than 1")
            return False
        
        self.volume_change_threshold = threshold
        self.logger.info(f"Set volume change threshold to {threshold}")
        return True
    
    def start_detection(self):
        """
        Start anomaly detection
        
        Returns:
            bool: Whether successfully started detection
        """
        if self.is_detecting:
            self.logger.warning("Anomaly detection is already running")
            return False
        
        if not self.data_streams:
            self.logger.error("No data streams to monitor")
            return False
        
        # Start detection thread
        self.is_detecting = True
        self.detection_thread = threading.Thread(target=self._detect_anomalies)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.logger.info("Started anomaly detection")
        return True
    
    def stop_detection(self):
        """
        Stop anomaly detection
        
        Returns:
            bool: Whether successfully stopped detection
        """
        if not self.is_detecting:
            self.logger.warning("Anomaly detection is not running")
            return False
        
        # Stop detection thread
        self.is_detecting = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)
        
        self.logger.info("Stopped anomaly detection")
        return True
    
    def _detect_anomalies(self):
        """
        Internal method to detect anomalies, runs in separate thread
        """
        while self.is_detecting:
            try:
                # Process each stream
                for stream_id in self.data_streams:
                    # Get data source
                    data_source = self.data_streams[stream_id]['data_source']
                    fields = self.data_streams[stream_id]['fields']
                    
                    # Get data from source
                    data = data_source()
                    
                    if data is None:
                        continue
                    
                    # Add to history
                    self.data_history[stream_id].append({
                        'time': datetime.now(),
                        'data': data
                    })
                    
                    # Limit history size
                    if len(self.data_history[stream_id]) > self.window_size * 2:
                        self.data_history[stream_id].pop(0)
                    
                    # Detect anomalies if we have enough data
                    if len(self.data_history[stream_id]) >= self.window_size:
                        # Get recent data for analysis
                        recent_data = self.data_history[stream_id][-self.window_size:]
                        
                        # Extract values for analysis
                        values = self._extract_values(recent_data, fields)
                        
                        # Detect anomalies
                        anomalies = self._detect_anomalies_in_values(stream_id, values, recent_data)
                        
                        # Update detection results
                        self.detection_results[stream_id] = {
                            'anomalies': anomalies,
                            'last_check': datetime.now()
                        }
                        
                        # Add to anomaly history
                        for anomaly in anomalies:
                            self.anomaly_history[stream_id].append(anomaly)
                        
                        # Limit anomaly history size
                        if len(self.anomaly_history[stream_id]) > 1000:
                            self.anomaly_history[stream_id] = self.anomaly_history[stream_id][-1000:]
                        
                        # Call callback if provided and anomalies were detected
                        if anomalies and stream_id in self.anomaly_callbacks:
                            try:
                                self.anomaly_callbacks[stream_id](anomalies)
                            except Exception as e:
                                self.logger.error(f"Error in anomaly callback for {stream_id}: {str(e)}")
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error detecting anomalies: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _extract_values(self, data_points, fields):
        """
        Extract values from data points for analysis
        
        Args:
            data_points (list): List of data points
            fields (list): List of fields to extract, None means extract all
            
        Returns:
            dict: Dictionary of field values
        """
        values = {}
        
        # Determine fields to extract
        if fields is None:
            # Get all fields from first data point
            if data_points and 'data' in data_points[0]:
                first_data = data_points[0]['data']
                if isinstance(first_data, dict):
                    fields = list(first_data.keys())
                else:
                    fields = ['value']
            else:
                fields = ['value']
        
        # Extract values for each field
        for field in fields:
            field_values = []
            
            for data_point in data_points:
                data = data_point['data']
                
                if isinstance(data, dict):
                    if field in data:
                        value = data[field]
                        # Convert to numeric if possible
                        try:
                            value = float(value)
                            field_values.append(value)
                        except (ValueError, TypeError):
                            pass
                else:
                    # Treat as single value
                    try:
                        value = float(data)
                        field_values.append(value)
                    except (ValueError, TypeError):
                        pass
            
            values[field] = field_values
        
        return values
    
    def _detect_anomalies_in_values(self, stream_id, values, data_points):
        """
        Detect anomalies in values
        
        Args:
            stream_id (str): Stream identifier
            values (dict): Dictionary of field values
            data_points (list): List of data points
            
        Returns:
            list: List of anomalies
        """
        anomalies = []
        
        # Get latest data point
        latest_data = data_points[-1]['data']
        latest_time = data_points[-1]['time']
        
        # Check each field
        for field, field_values in values.items():
            if not field_values or len(field_values) < 2:
                continue
            
            # Get latest value
            if isinstance(latest_data, dict):
                if field not in latest_data:
                    continue
                latest_value = latest_data[field]
                try:
                    latest_value = float(latest_value)
                except (ValueError, TypeError):
                    continue
            else:
                try:
                    latest_value = float(latest_data)
                except (ValueError, TypeError):
                    continue
            
            # Previous value
            prev_value = field_values[-2]
            
            # Detect sudden price changes
            if 'price' in field.lower() or 'close' in field.lower() or 'open' in field.lower() or 'high' in field.lower() or 'low' in field.lower():
                if prev_value != 0:
                    price_change = abs(latest_value - prev_value) / prev_value
                    if price_change > self.price_change_threshold:
                        anomalies.append({
                            'time': latest_time,
                            'field': field,
                            'value': latest_value,
                            'prev_value': prev_value,
                            'type': 'price_change',
                            'severity': 'high' if price_change > self.price_change_threshold * 2 else 'medium',
                            'description': f"Sudden price change detected: {price_change:.2%}"
                        })
            
            # Detect sudden volume changes
            if 'volume' in field.lower():
                if prev_value != 0:
                    volume_change = latest_value / prev_value
                    if volume_change > self.volume_change_threshold:
                        anomalies.append({
                            'time': latest_time,
                            'field': field,
                            'value': latest_value,
                            'prev_value': prev_value,
                            'type': 'volume_change',
                            'severity': 'high' if volume_change > self.volume_change_threshold * 2 else 'medium',
                            'description': f"Sudden volume change detected: {volume_change:.1f}x"
                        })
            
            # Detect frozen prices (no change over time)
            if 'price' in field.lower() or 'close' in field.lower():
                # Check if all recent values are the same
                if len(set(field_values[-10:])) == 1 and len(field_values) >= 10:
                    anomalies.append({
                        'time': latest_time,
                        'field': field,
                        'value': latest_value,
                        'type': 'frozen_price',
                        'severity': 'medium',
                        'description': "Frozen price detected (no change over time)"
                    })
            
            # Detect gaps in data
            if len(data_points) >= 2:
                # Check time gap
                time_gap = (data_points[-1]['time'] - data_points[-2]['time']).total_seconds()
                if time_gap > 60:  # More than a minute gap
                    anomalies.append({
                        'time': latest_time,
                        'field': field,
                        'value': latest_value,
                        'type': 'data_gap',
                        'severity': 'medium',
                        'description': f"Data gap detected: {time_gap} seconds"
                    })
            
            # Statistical anomaly detection if we have enough data
            if len(field_values) >= 10:
                # Z-score method
                z_score = self._calculate_z_score(latest_value, field_values)
                if abs(z_score) > self.z_score_threshold:
                    anomalies.append({
                        'time': latest_time,
                        'field': field,
                        'value': latest_value,
                        'type': 'statistical',
                        'method': 'z_score',
                        'z_score': z_score,
                        'severity': 'high' if abs(z_score) > self.z_score_threshold * 1.5 else 'medium',
                        'description': f"Statistical anomaly detected (Z-score: {z_score:.2f})"
                    })
                
                # IQR method
                iqr_outlier = self._detect_iqr_outlier(latest_value, field_values)
                if iqr_outlier:
                    anomalies.append({
                        'time': latest_time,
                        'field': field,
                        'value': latest_value,
                        'type': 'statistical',
                        'method': 'iqr',
                        'severity': 'medium',
                        'description': "Statistical anomaly detected (IQR method)"
                    })
            
            # Machine learning method if we have enough data
            if len(field_values) >= self.window_size:
                # Train or update model
                self._update_model(stream_id, field, field_values)
                
                # Predict anomaly
                if self.models[stream_id] is not None and field in self.models[stream_id]:
                    model = self.models[stream_id][field]
                    
                    # Prepare data for prediction
                    X = np.array(field_values[-self.window_size:]).reshape(-1, 1)
                    
                    # Predict
                    anomaly_score = model.decision_function(X)[-1]
                    is_anomaly = model.predict(X)[-1] == -1
                    
                    if is_anomaly:
                        anomalies.append({
                            'time': latest_time,
                            'field': field,
                            'value': latest_value,
                            'type': 'statistical',
                            'method': 'isolation_forest',
                            'anomaly_score': anomaly_score,
                            'severity': 'high',
                            'description': f"Statistical anomaly detected (Isolation Forest, score: {anomaly_score:.2f})"
                        })
        
        return anomalies
    
    def _calculate_z_score(self, value, values):
        """
        Calculate Z-score for a value
        
        Args:
            value (float): Value to calculate Z-score for
            values (list): List of values for comparison
            
        Returns:
            float: Z-score
        """
        if len(values) < 2:
            return 0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0
        
        return (value - mean) / std
    
    def _detect_iqr_outlier(self, value, values):
        """
        Detect outlier using IQR method
        
        Args:
            value (float): Value to check
            values (list): List of values for comparison
            
        Returns:
            bool: Whether value is an outlier
        """
        if len(values) < 4:
            return False
        
        # Calculate IQR
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        # Calculate bounds
        lower_bound = q1 - self.iqr_factor * iqr
        upper_bound = q3 + self.iqr_factor * iqr
        
        # Check if value is outside bounds
        return value < lower_bound or value > upper_bound
    
    def _update_model(self, stream_id, field, values):
        """
        Update or train Isolation Forest model
        
        Args:
            stream_id (str): Stream identifier
            field (str): Field name
            values (list): List of values
        """
        # Initialize models dictionary if needed
        if self.models[stream_id] is None:
            self.models[stream_id] = {}
        
        # Check if we need to train or update the model
        if field not in self.models[stream_id] or len(values) % self.window_size == 0:
            try:
                # Prepare data
                X = np.array(values[-self.window_size:]).reshape(-1, 1)
                
                # Create and train model
                model = IsolationForest(
                    contamination=self.contamination,
                    random_state=42
                )
                model.fit(X)
                
                # Store model
                self.models[stream_id][field] = model
                
                self.logger.debug(f"Updated Isolation Forest model for {stream_id}.{field}")
            except Exception as e:
                self.logger.error(f"Error updating Isolation Forest model for {stream_id}.{field}: {str(e)}")
    
    def get_detection_result(self, stream_id):
        """
        Get detection result for a stream
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            dict: Detection result or None if stream not found
        """
        if stream_id not in self.detection_results:
            self.logger.warning(f"Detection results for {stream_id} not found")
            return None
        
        return self.detection_results[stream_id].copy()
    
    def get_anomaly_history(self, stream_id, count=None):
        """
        Get anomaly history for a stream
        
        Args:
            stream_id (str): Stream identifier
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of anomaly records or None if stream not found
        """
        if stream_id not in self.anomaly_history:
            self.logger.warning(f"Anomaly history for {stream_id} not found")
            return None
        
        if count is None:
            return self.anomaly_history[stream_id].copy()
        else:
            return self.anomaly_history[stream_id][-count:]
    
    def get_anomaly_history_dataframe(self, stream_id, count=None):
        """
        Get anomaly history for a stream as DataFrame
        
        Args:
            stream_id (str): Stream identifier
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing anomaly history or None if stream not found
        """
        history = self.get_anomaly_history(stream_id, count)
        
        if history is None:
            return None
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    def save_anomaly_history_to_csv(self, stream_id, filename, count=None):
        """
        Save anomaly history for a stream to CSV file
        
        Args:
            stream_id (str): Stream identifier
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_anomaly_history_dataframe(stream_id, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No anomaly history to save for stream {stream_id}")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} anomaly records for {stream_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save anomaly history for {stream_id} to CSV: {str(e)}")
            return False
    
    def save_anomaly_history_to_parquet(self, stream_id, filename, count=None):
        """
        Save anomaly history for a stream to Parquet file
        
        Args:
            stream_id (str): Stream identifier
            filename (str): Filename
            count (int): Number of records to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_anomaly_history_dataframe(stream_id, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No anomaly history to save for stream {stream_id}")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} anomaly records for {stream_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save anomaly history for {stream_id} to Parquet: {str(e)}")
            return False
    
    def reset_model(self, stream_id, field=None):
        """
        Reset model for a stream or field
        
        Args:
            stream_id (str): Stream identifier
            field (str): Field name, None means reset all fields
            
        Returns:
            bool: Whether successfully reset
        """
        if stream_id not in self.models:
            self.logger.warning(f"Models for {stream_id} not found")
            return False
        
        if field is None:
            # Reset all models for stream
            self.models[stream_id] = {}
            self.logger.info(f"Reset all models for stream {stream_id}")
        else:
            # Reset specific field model
            if self.models[stream_id] is not None and field in self.models[stream_id]:
                del self.models[stream_id][field]
                self.logger.info(f"Reset model for {stream_id}.{field}")
            else:
                self.logger.warning(f"Model for {stream_id}.{field} not found")
                return False
        
        return True
