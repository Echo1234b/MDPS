import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class DriftAlerting:
    def __init__(self, config_path: str = None, log_level: str = "INFO"):
        """
        Initialize the Drift Alerting system.
        
        Args:
            config_path: Path to the configuration file
            log_level: Logging level
        """
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("DriftAlerting")
        
        # Default configuration
        self.default_config = {
            "drift_threshold": 0.1,
            "warning_threshold": 0.05,
            "check_interval_minutes": 60,
            "window_size_days": 30,
            "notification_email": None,
            "notification_webhook": None,
            "models": {},
            "drift_methods": {
                "statistical": True,
                "distribution": True,
                "pca": True,
                "isolation_forest": True,
                "custom": False
            }
        }
        
        # Load configuration
        self.config_path = config_path
        if config_path and os.path.exists(config_path):
            self.config = self._load_config()
        else:
            self.config = self.default_config.copy()
            if config_path:
                self._save_config()
        
        # Model registry
        self.model_registry = {}
        
        # Drift history
        self.drift_history = []
        
        # Reference data storage
        self.reference_data = {}
        
        # Custom drift detection methods
        self.custom_drift_methods = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def add_model(self, model_name: str, data_source: str, feature_columns: List[str],
                  target_column: str = None, drift_threshold: float = None,
                  warning_threshold: float = None, check_interval_minutes: int = None,
                  window_size_days: int = None
    def add_model(self, model_name: str, data_source: str, feature_columns: List[str],
                  target_column: str = None, drift_threshold: float = None,
                  warning_threshold: float = None, check_interval_minutes: int = None,
                  window_size_days: int = None, custom_drift_method: Callable = None):
        """
        Add a model to the drift alerting system.
        
        Args:
            model_name: Name of the model
            data_source: Path to the data source
            feature_columns: List of feature column names
            target_column: Name of the target column (optional)
            drift_threshold: Drift threshold (overrides default)
            warning_threshold: Warning threshold (overrides default)
            check_interval_minutes: Check interval in minutes (overrides default)
            window_size_days: Window size in days for reference data (overrides default)
            custom_drift_method: Custom drift detection method
        """
        # Add model to configuration
        self.config["models"][model_name] = {
            "data_source": data_source,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "drift_threshold": drift_threshold if drift_threshold is not None else self.config["drift_threshold"],
            "warning_threshold": warning_threshold if warning_threshold is not None else self.config["warning_threshold"],
            "check_interval_minutes": check_interval_minutes if check_interval_minutes is not None else self.config["check_interval_minutes"],
            "window_size_days": window_size_days if window_size_days is not None else self.config["window_size_days"],
            "last_checked": datetime.now().isoformat(),
            "active": True
        }
        
        # Save configuration
        self._save_config()
        
        # Add to model registry
        self.model_registry[model_name] = {
            "custom_drift_method": custom_drift_method
        }
        
        # Store custom drift method if provided
        if custom_drift_method:
            self.custom_drift_methods[model_name] = custom_drift_method
            self.config["drift_methods"]["custom"] = True
            self._save_config()
        
        # Load reference data
        self._load_reference_data(model_name)
        
        self.logger.info(f"Added model '{model_name}' to drift alerting system")
    
    def remove_model(self, model_name: str):
        """
        Remove a model from the drift alerting system.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in drift alerting system")
            return
        
        # Remove from configuration
        del self.config["models"][model_name]
        self._save_config()
        
        # Remove from model registry
        if model_name in self.model_registry:
            del self.model_registry[model_name]
        
        # Remove custom drift method if exists
        if model_name in self.custom_drift_methods:
            del self.custom_drift_methods[model_name]
        
        # Remove reference data
        if model_name in self.reference_data:
            del self.reference_data[model_name]
        
        self.logger.info(f"Removed model '{model_name}' from drift alerting system")
    
    def update_model(self, model_name: str, **kwargs):
        """
        Update a model's configuration.
        
        Args:
            model_name: Name of the model
            **kwargs: Configuration parameters to update
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in drift alerting system")
            return
        
        # Update configuration
        for key, value in kwargs.items():
            if key in self.config["models"][model_name]:
                self.config["models"][model_name][key] = value
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")
        
        # Save configuration
        self._save_config()
        
        # Reload reference data if data source changed
        if "data_source" in kwargs or "feature_columns" in kwargs:
            self._load_reference_data(model_name)
        
        self.logger.info(f"Updated configuration for model '{model_name}'")
    
    def _load_reference_data(self, model_name: str):
        """
        Load reference data for a model.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in drift alerting system")
            return
        
        model_config = self.config["models"][model_name]
        data_source = model_config["data_source"]
        feature_columns = model_config["feature_columns"]
        target_column = model_config.get("target_column")
        
        try:
            # Load data
            if data_source.endswith('.csv'):
                data = pd.read_csv(data_source)
            elif data_source.endswith('.parquet'):
                data = pd.read_parquet(data_source)
            else:
                raise ValueError(f"Unsupported data source format: {data_source}")
            
            # Filter by window size
            window_size_days = model_config["window_size_days"]
            if window_size_days > 0:
                cutoff_date = datetime.now() - timedelta(days=window_size_days)
                
                # Try to find a date column
                date_columns = [col for col in data.columns if 'date' in col.lower()]
                if date_columns:
                    date_column = date_columns[0]
                    data[date_column] = pd.to_datetime(data[date_column])
                    data = data[data[date_column] >= cutoff_date]
            
            # Select features and target
            columns_to_use = feature_columns.copy()
            if target_column and target_column in data.columns:
                columns_to_use.append(target_column)
            
            reference_data = data[columns_to_use].copy()
            
            # Store reference data
            self.reference_data[model_name] = reference_data
            
            self.logger.info(f"Loaded reference data for model '{model_name}': {reference_data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading reference data for model '{model_name}': {str(e)}")
    
    def check_drift(self, model_name: str = None, current_data: pd.DataFrame = None):
        """
        Check for drift in model data.
        
        Args:
            model_name: Name of the model (if None, check all models)
            current_data: Current data to compare with reference data (if None, load from data source)
            
        Returns:
            Drift detection results
        """
        if model_name:
            models_to_check = [model_name]
        else:
            models_to_check = list(self.config["models"].keys())
        
        results = {}
        
        for name in models_to_check:
            if name not in self.config["models"]:
                self.logger.warning(f"Model '{name}' not found in drift alerting system")
                continue
            
            model_config = self.config["models"][name]
            
            # Check if model is active
            if not model_config.get("active", True):
                self.logger.info(f"Model '{name}' is inactive, skipping drift check")
                continue
            
            # Get current data
            if current_data is None:
                try:
                    data_source = model_config["data_source"]
                    feature_columns = model_config["feature_columns"]
                    target_column = model_config.get("target_column")
                    
                    # Load data
                    if data_source.endswith('.csv'):
                        data = pd.read_csv(data_source)
                    elif data_source.endswith('.parquet'):
                        data = pd.read_parquet(data_source)
                    else:
                        raise ValueError(f"Unsupported data source format: {data_source}")
                    
                    # Select features and target
                    columns_to_use = feature_columns.copy()
                    if target_column and target_column in data.columns:
                        columns_to_use.append(target_column)
                    
                    current_data = data[columns_to_use].copy()
                    
                except Exception as e:
                    self.logger.error(f"Error loading current data for model '{name}': {str(e)}")
                    continue
            
            # Check if reference data exists
            if name not in self.reference_data:
                self.logger.warning(f"No reference data for model '{name}'")
                continue
            
            reference_data = self.reference_data[name]
            
            # Perform drift detection
            drift_scores = {}
            drift_detected = False
            warning_triggered = False
            
            # Statistical drift detection
            if self.config["drift_methods"]["statistical"]:
                try:
                    statistical_score = self._detect_statistical_drift(reference_data, current_data)
                    drift_scores["statistical"] = statistical_score
                    
                    if statistical_score > model_config["drift_threshold"]:
                        drift_detected = True
                    elif statistical_score > model_config["warning_threshold"]:
                        warning_triggered = True
                        
                except Exception as e:
                    self.logger.error(f"Error in statistical drift detection for model '{name}': {str(e)}")
            
            # Distribution drift detection
            if self.config["drift_methods"]["distribution"]:
                try:
                    distribution_score = self._detect_distribution_drift(reference_data, current_data)
                    drift_scores["distribution"] = distribution_score
                    
                    if distribution_score > model_config["drift_threshold"]:
                        drift_detected = True
                    elif distribution_score > model_config["warning_threshold"]:
                        warning_triggered = True
                        
                except Exception as e:
                    self.logger.error(f"Error in distribution drift detection for model '{name}': {str(e)}")
            
            # PCA drift detection
            if self.config["drift_methods"]["pca"]:
                try:
                    pca_score = self._detect_pca_drift(reference_data, current_data)
                    drift_scores["pca"] = pca_score
                    
                    if pca_score > model_config["drift_threshold"]:
                        drift_detected = True
                    elif pca_score > model_config["warning_threshold"]:
                        warning_triggered = True
                        
                except Exception as e:
                    self.logger.error(f"Error in PCA drift detection for model '{name}': {str(e)}")
            
            # Isolation Forest drift detection
            if self.config["drift_methods"]["isolation_forest"]:
                try:
                    isolation_score = self._detect_isolation_drift(reference_data, current_data)
                    drift_scores["isolation_forest"] = isolation_score
                    
                    if isolation_score > model_config["drift_threshold"]:
                        drift_detected = True
                    elif isolation_score > model_config["warning_threshold"]:
                        warning_triggered = True
                        
                except Exception as e:
                    self.logger.error(f"Error in Isolation Forest drift detection for model '{name}': {str(e)}")
            
            # Custom drift detection
            if self.config["drift_methods"]["custom"] and name in self.custom_drift_methods:
                try:
                    custom_score = self.custom_drift_methods[name](reference_data,%20current_data)
                    drift_scores["custom"] = custom_score
                    
                    if custom_score > model_config["drift_threshold"]:
                        drift_detected = True
                    elif custom_score > model_config["warning_threshold"]:
                        warning_triggered = True
                        
                except Exception as e:
                    self.logger.error(f"Error in custom drift detection for model '{name}': {str(e)}")
            
            # Calculate overall drift score
            if drift_scores:
                overall_score = np.mean(list(drift_scores.values()))
            else:
                overall_score = 0.0
            
            # Store results
            results[name] = {
                "timestamp": datetime.now().isoformat(),
                "drift_scores": drift_scores,
                "overall_score": overall_score,
                "drift_detected": drift_detected,
                "warning_triggered": warning_triggered
            }
            
            # Add to history
            self.drift_history.append({
                "model_name": name,
                "timestamp": datetime.now().isoformat(),
                "drift_scores": drift_scores,
                "overall_score": overall_score,
                "drift_detected": drift_detected,
                "warning_triggered": warning_triggered
            })
            
            # Update last checked time
            model_config["last_checked"] = datetime.now().isoformat()
            self._save_config()
            
            # Send notification if drift detected
            if drift_detected:
                self._send_notification(
                    f"Drift detected in model '{name}' (score: {overall_score:.4f})",
                    level="alert"
                )
            elif warning_triggered:
                self._send_notification(
                    f"Drift warning for model '{name}' (score: {overall_score:.4f})",
                    level="warning"
                )
            
            self.logger.info(f"Drift check completed for model '{name}': score={overall_score:.4f}, drift={drift_detected}")
        
        return results
    
    def _detect_statistical_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> float:
        """
        Detect drift using statistical tests.
        
        Args:
            reference_data: Reference data
            current_data: Current data
            
        Returns:
            Drift score
        """
        # Get common columns
        common_columns = [col for col in reference_data.columns if col in current_data.columns]
        
        if not common_columns:
            return 0.0
        
        drift_scores = []
        
        for column in common_columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(reference_data[column]):
                continue
            
            ref_values = reference_data[column].dropna()
            curr_values = current_data[column].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            # Perform Kolmogorov-Smirnov test
            try:
                ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
                drift_scores.append(ks_stat)
            except:
                continue
        
        # Return mean drift score
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _detect_distribution_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> float:
        """
        Detect drift by comparing distributions.
        
        Args:
            reference_data: Reference data
            current_data: Current data
            
        Returns:
            Drift score
        """
        # Get common columns
        common_columns = [col for col in reference_data.columns if col in current_data.columns]
        
        if not common_columns:
            return 0.0
        
        drift_scores = []
        
        for column in common_columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(reference_data[column]):
                continue
            
            ref_values = reference_data[column].dropna()
            curr_values = current_data[column].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            # Calculate histogram bins
            min_val = min(ref_values.min(), curr_values.min())
            max_val = max(ref_values.max(), curr_values.max())
            
            # Use 10 bins
            bins = np.linspace(min_val, max_val, 11)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(ref_values, bins=bins)
            curr_hist, _ = np.histogram(curr_values, bins=bins)
            
            # Normalize histograms
            ref_hist = ref_hist / ref_hist.sum()
            curr_hist = curr_hist / curr_hist.sum()
            
            # Calculate Jensen-Shannon divergence
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            curr_hist = curr_hist + epsilon
            
            avg_hist = 0.5 * (ref_hist + curr_hist)
            
            kl_ref = np.sum(ref_hist * np.log(ref_hist / avg_hist))
            kl_curr = np.sum(curr_hist * np.log(curr_hist / avg_hist))
            
            js_divergence = 0.5 * (kl_ref + kl_curr)
            
            drift_scores.append(js_divergence)
        
        # Return mean drift score
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _detect_pca_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> float:
        """
        Detect drift using PCA.
        
        Args:
            reference_data: Reference data
            current_data: Current data
            
        Returns:
            Drift score
        """
        # Get common numeric columns
        common_columns = [
            col for col in reference_data.columns 
            if col in current_data.columns and pd.api.types.is_numeric_dtype(reference_data[col])
        ]
        
        if not common_columns:
            return 0.0
        
        # Prepare data
        ref_data = reference_data[common_columns].dropna()
        curr_data = current_data[common_columns].dropna()
        
        if len(ref_data) == 0 or len(curr_data) == 0:
            return 0.0
        
        # Standardize data
        scaler = StandardScaler()
        ref_scaled = scaler.fit_transform(ref_data)
        curr_scaled = scaler.transform(curr_data)
        
        # Fit PCA on reference data
        n_components = min(5, len(common_columns))
        pca = PCA(n_components=n_components)
        pca.fit(ref_scaled)
        
        # Transform both datasets
        ref_pca = pca.transform(ref_scaled)
        curr_pca = pca.transform(curr_scaled)
        
        # Calculate mean squared distance between PCA projections
        distances = []
        
        # Sample if datasets are large
        max_samples = 1000
        ref_sample_size = min(len(ref_pca), max_samples)
        curr_sample_size = min(len(curr_pca), max_samples)
        
        ref_indices = np.random.choice(len(ref_pca), ref_sample_size, replace=False)
        curr_indices = np.random.choice(len(curr_pca), curr_sample_size, replace=False)
        
        ref_sample = ref_pca[ref_indices]
        curr_sample = curr_pca[curr_indices]
        
        for i in range(ref_sample_size):
            for j in range(curr_sample_size):
                distance = np.sum((ref_sample[i] - curr_sample[j]) ** 2)
                distances.append(distance)
        
        # Return mean distance as drift score
        return np.mean(distances) if distances else 0.0
    
    def _detect_isolation_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> float:
        """
        Detect drift using Isolation Forest.
        
        Args:
            reference_data: Reference data
            current_data: Current data
            
        Returns:
            Drift score
        """
        # Get common numeric columns
        common_columns = [
            col for col in reference_data.columns 
            if col in current_data.columns and pd.api.types.is_numeric_dtype(reference_data[col])
        ]
        
        if not common_columns:
            return 0.0
        
        # Prepare data
        ref_data = reference_data[common_columns].dropna()
        curr_data = current_data[common_columns].dropna()
        
        if len(ref_data) == 0 or len(curr_data) == 0:
            return 0.0
        
        # Standardize data
        scaler = StandardScaler()
        ref_scaled = scaler.fit_transform(ref_data)
        curr_scaled = scaler.transform(curr_data)
        
        # Fit Isolation Forest on reference data
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(ref_scaled)
        
        # Predict on current data
        curr_predictions = iso_forest.predict(curr_scaled)
        
        # Calculate proportion of outliers
        outlier_ratio = np.mean(curr_predictions == -1)
        
        return outlier_ratio
    
    def _send_notification(self, message: str, level: str = "info"):
        """
        Send a notification.
        
        Args:
            message: Notification message
            level: Notification level
        """
        # Log the message
        if level == "alert":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # Send email if configured
        if self.config["notification_email"]:
            self._send_email_notification(message, level)
        
        # Send webhook if configured
        if self.config["notification_webhook"]:
            self._send_webhook_notification(message, level)
    
    def _send_email_notification(self, message: str, level: str):
        """
        Send an email notification.
        
        Args:
            message: Notification message
            level: Notification level
        """
        # This is a placeholder implementation
        # In a real implementation, this would send an email
        self.logger.info(f"Email notification ({level}): {message}")
    
    def _send_webhook_notification(self, message: str, level: str):
        """
        Send a webhook notification.
        
        Args:
            message: Notification message
            level: Notification level
        """
        # This is a placeholder implementation
        # In a real implementation, this would send a webhook
        self.logger.info(f"Webhook notification ({level}): {message}")
    
    def get_drift_history(self, model_name: str = None) -> pd.DataFrame:
        """
        Get drift history.
        
        Args:
            model_name: Name of the model (if None, return history for all models)
            
        Returns:
            DataFrame with drift history
        """
        history = self.drift_history
        
        if model_name:
            history = [entry for entry in history if entry["model_name"] == model_name]
        
        return pd.DataFrame(history)
    
    def get_model_status(self, model_name: str = None) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Get the status of models.
        
        Args:
            model_name: Name of the model (if None, return status for all models)
            
        Returns:
            Model status or DataFrame with status for all models
        """
        if model_name:
            if model_name not in self.config["models"]:
                raise ValueError(f"Model '{model_name}' not found in drift alerting system")
            
            return self.config["models"][model_name].copy()
        else:
            status_data = []
            
            for name, config in self.config["models"].items():
                # Get latest drift score
                latest_drift = None
                for entry in reversed(self.drift_history):
                    if entry["model_name"] == name:
                        latest_drift = entry
                        break
                
                status_data.append({
                    "model_name": name,
                    "active": config.get("active", True),
                    "last_checked": config["last_checked"],
                    "latest_drift_score": latest_drift["overall_score"] if latest_drift else None,
                    "drift_detected": latest_drift["drift_detected"] if latest_drift else False,
                    "warning_triggered": latest_drift["warning_triggered"] if latest_drift else False
                })
            
            return pd.DataFrame(status_data)
    
    def save_drift_alerting(self, filepath: str):
        """
        Save the drift alerting system to a file.
        
        Args:
            filepath: Path to save the system
        """
        # Create a dictionary of all attributes
        drift_alerting_data = {
            "config": self.config,
            "drift_history": self.drift_history,
            "reference_data": {
                name: data.to_dict() for name, data in self.reference_data.items()
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(drift_alerting_data, f)
    
    def load_drift_alerting(self, filepath: str):
        """
        Load a drift alerting system from a file.
        
        Args:
            filepath: Path to the system file
        """
        # Load from file
        with open(filepath, 'r') as f:
            drift_alerting_data = json.load(f)
        
        # Set attributes
        self.config = drift_alerting_data["config"]
        self.drift_history = drift_alerting_data["drift_history"]
        
        # Load reference data
        self.reference_data = {}
        for name, data_dict in drift_alerting_data["reference_data"].items():
            self.reference_data[name] = pd.DataFrame.from_dict(data_dict)
        
        return self
