import os
import json
import time
import schedule
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class RetrainingScheduler:
    def __init__(self, config_path: str = None, log_level: str = "INFO"):
        """
        Initialize the Retraining Scheduler.
        
        Args:
            config_path: Path to the configuration file
            log_level: Logging level
        """
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RetrainingScheduler")
        
        # Default configuration
        self.default_config = {
            "retraining_interval_days": 7,
            "performance_threshold": 0.8,
            "data_drift_threshold": 0.1,
            "max_retraining_attempts": 3,
            "notification_email": None,
            "notification_webhook": None,
            "backup_before_retraining": True,
            "backup_dir": "./model_backups",
            "models": {}
        }
        
        # Load configuration
        self.config_path = config_path
        if config_path and os.path.exists(config_path):
            self.config = self._load_config()
        else:
            self.config = self.default_config.copy()
            if config_path:
                self._save_config()
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.config["backup_dir"], exist_ok=True)
        
        # Initialize scheduler
        self.scheduler = schedule.Scheduler()
        self.scheduler_thread = None
        self.running = False
        
        # Model registry
        self.model_registry = {}
        
        # Retraining history
        self.retraining_history = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def add_model(self, model_name: str, model_path: str, data_source: str, 
                  target_column: str, retraining_interval_days: int = None,
                  performance_threshold: float = None, data_drift_threshold: float = None,
                  max_retraining_attempts: int = None, custom_retraining_condition: Callable = None):
        """
        Add a model to the retraining scheduler.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            data_source: Path to the data source
            target_column: Name of the target column
            retraining_interval_days: Interval in days for retraining (overrides default)
            performance_threshold: Performance threshold for retraining (overrides default)
            data_drift_threshold: Data drift threshold for retraining (overrides default)
            max_retraining_attempts: Maximum number of retraining attempts (overrides default)
            custom_retraining_condition: Custom function to determine if retraining is needed
        """
        # Add model to configuration
        self.config["models"][model_name] = {
            "model_path": model_path,
            "data_source": data_source,
            "target_column": target_column,
            "retraining_interval_days": retraining_interval_days if retraining_interval_days is not None else self.config["retraining_interval_days"],
            "performance_threshold": performance_threshold if performance_threshold is not None else self.config["performance_threshold"],
            "data_drift_threshold": data_drift_threshold if data_drift_threshold is not None else self.config["data_drift_threshold"],
            "max_retraining_attempts": max_retraining_attempts if max_retraining_attempts is not None else self.config["max_retraining_attempts"],
            "last_retrained": datetime.now().isoformat(),
            "retraining_count": 0,
            "retraining_attempts": 0,
            "active": True
        }
        
        # Save configuration
        self._save_config()
        
        # Add to model registry
        self.model_registry[model_name] = {
            "custom_retraining_condition": custom_retraining_condition
        }
        
        # Schedule retraining
        self._schedule_retraining(model_name)
        
        self.logger.info(f"Added model '{model_name}' to retraining scheduler")
    
    def remove_model(self, model_name: str):
        """
        Remove a model from the retraining scheduler.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in scheduler")
            return
        
        # Remove from configuration
        del self.config["models"][model_name]
        self._save_config()
        
        # Remove from model registry
        if model_name in self.model_registry:
            del self.model_registry[model_name]
        
        # Unschedule retraining
        schedule.clear(model_name)
        
        self.logger.info(f"Removed model '{model_name}' from retraining scheduler")
    
    def update_model(self, model_name: str, **kwargs):
        """
        Update a model's configuration.
        
        Args:
            model_name: Name of the model
            **kwargs: Configuration parameters to update
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in scheduler")
            return
        
        # Update configuration
        for key, value in kwargs.items():
            if key in self.config["models"][model_name]:
                self.config["models"][model_name][key] = value
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")
        
        # Save configuration
        self._save_config()
        
        # Reschedule retraining if interval changed
        if "retraining_interval_days" in kwargs:
            schedule.clear(model_name)
            self._schedule_retraining(model_name)
        
        self.logger.info(f"Updated configuration for model '{model_name}'")
    
    def _schedule_retraining(self, model_name: str):
        """
        Schedule retraining for a model.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in scheduler")
            return
        
        model_config = self.config["models"][model_name]
        interval_days = model_config["retraining_interval_days"]
        
        # Schedule retraining
        self.scheduler.every(interval_days).days.do(
            self._check_and_retrain, model_name
        ).tag(model_name)
        
        self.logger.info(f"Scheduled retraining for model '{model_name}' every {interval_days} days")
    
    def _check_and_retrain(self, model_name: str):
        """
        Check if a model needs retraining and retrain if necessary.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in scheduler")
            return
        
        model_config = self.config["models"][model_name]
        
        # Check if model is active
        if not model_config.get("active", True):
            self.logger.info(f"Model '{model_name}' is inactive, skipping retraining")
            return
        
        # Check if retraining is needed
        retraining_needed = False
        reason = None
        
        # Check if custom condition is provided
        if model_name in self.model_registry and self.model_registry[model_name].get("custom_retraining_condition"):
            try:
                custom_condition = self.model_registry[model_name]["custom_retraining_condition"]
                if custom_condition(model_name):
                    retraining_needed = True
                    reason = "Custom condition met"
            except Exception as e:
                self.logger.error(f"Error checking custom condition for model '{model_name}': {str(e)}")
        
        # Check if retraining interval has passed
        if not retraining_needed:
            last_retrained = datetime.fromisoformat(model_config["last_retrained"])
            if datetime.now() - last_retrained >= timedelta(days=model_config["retraining_interval_days"]):
                retraining_needed = True
                reason = "Retraining interval passed"
        
        # Check performance threshold
        if not retraining_needed:
            try:
                performance = self._evaluate_model_performance(model_name)
                if performance < model_config["performance_threshold"]:
                    retraining_needed = True
                    reason = f"Performance ({performance}) below threshold ({model_config['performance_threshold']})"
            except Exception as e:
                self.logger.error(f"Error evaluating performance for model '{model_name}': {str(e)}")
        
        # Check data drift
        if not retraining_needed:
            try:
                drift_score = self._calculate_data_drift(model_name)
                if drift_score > model_config["data_drift_threshold"]:
                    retraining_needed = True
                    reason = f"Data drift ({drift_score}) above threshold ({model_config['data_drift_threshold']})"
            except Exception as e:
                self.logger.error(f"Error calculating data drift for model '{model_name}': {str(e)}")
        
        # Retrain if needed
        if retraining_needed:
            self.logger.info(f"Retraining model '{model_name}' because: {reason}")
            self._retrain_model(model_name)
        else:
            self.logger.info(f"No retraining needed for model '{model_name}'")
    
    def _evaluate_model_performance(self, model_name: str) -> float:
        """
        Evaluate the performance of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Performance score
        """
        # This is a placeholder implementation
        # In a real implementation, this would load the model and data,
        # make predictions, and calculate performance metrics
        
        # For now, return a random value
        return np.random.random()
    
    def _calculate_data_drift(self, model_name: str) -> float:
        """
        Calculate data drift for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Data drift score
        """
        # This is a placeholder implementation
        # In a real implementation, this would compare the distribution
        # of the current data with the training data
        
        # For now, return a random value
        return np.random.random()
    
    def _retrain_model(self, model_name: str):
        """
        Retrain a model.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in scheduler")
            return
        
        model_config = self.config["models"][model_name]
        
        # Check if maximum attempts reached
        if model_config["retraining_attempts"] >= model_config["max_retraining_attempts"]:
            self.logger.warning(f"Maximum retraining attempts reached for model '{model_name}'")
            self._send_notification(
                f"Maximum retraining attempts reached for model '{model_name}'",
                level="warning"
            )
            return
        
        # Backup model if configured
        if self.config["backup_before_retraining"]:
            self._backup_model(model_name)
        
        # Increment attempts
        model_config["retraining_attempts"] += 1
        self._save_config()
        
        try:
            # This is a placeholder implementation
            # In a real implementation, this would load the data,
            # retrain the model, and save the updated model
            
            # Simulate retraining
            self.logger.info(f"Retraining model '{model_name}' (attempt {model_config['retraining_attempts']})")
            time.sleep(2)  # Simulate training time
            
            # Update configuration
            model_config["last_retrained"] = datetime.now().isoformat()
            model_config["retraining_count"] += 1
            model_config["retraining_attempts"] = 0  # Reset attempts on success
            self._save_config()
            
            # Add to history
            self.retraining_history.append({
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "attempt": model_config["retraining_count"]
            })
            
            self.logger.info(f"Successfully retrained model '{model_name}'")
            self._send_notification(f"Successfully retrained model '{model_name}'")
            
        except Exception as e:
            self.logger.error(f"Error retraining model '{model_name}': {str(e)}")
            
            # Add to history
            self.retraining_history.append({
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "attempt": model_config["retraining_attempts"],
                "error": str(e)
            })
            
            self._send_notification(
                f"Error retraining model '{model_name}': {str(e)}",
                level="error"
            )
    
    def _backup_model(self, model_name: str):
        """
        Backup a model before retraining.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.config["models"]:
            self.logger.warning(f"Model '{model_name}' not found in scheduler")
            return
        
        model_config = self.config["models"][model_name]
        model_path = model_config["model_path"]
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            return
        
        # Create backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{model_name}_{timestamp}.pkl"
        backup_path = os.path.join(self.config["backup_dir"], backup_filename)
        
        # Copy model file
        import shutil
        shutil.copy2(model_path, backup_path)
        
        self.logger.info(f"Backed up model '{model_name}' to {backup_path}")
    
    def _send_notification(self, message: str, level: str = "info"):
        """
        Send a notification.
        
        Args:
            message: Notification message
            level: Notification level
        """
        # Log the message
        if level == "error":
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
    
    def start(self):
        """Start the retraining scheduler."""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.logger.info("Started retraining scheduler")
    
    def stop(self):
        """Stop the retraining scheduler."""
        if not self.running:
            self.logger.warning("Scheduler is not running")
            return
        
        self.running = False
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Stopped retraining scheduler")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.running:
            self.scheduler.run_pending()
            time.sleep(1)
    
    def get_retraining_history(self, model_name: str = None) -> pd.DataFrame:
        """
        Get retraining history.
        
        Args:
            model_name: Name of the model (if None, return history for all models)
            
        Returns:
            DataFrame with retraining history
        """
        history = self.retraining_history
        
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
                raise ValueError(f"Model '{model_name}' not found in scheduler")
            
            return self.config["models"][model_name].copy()
        else:
            status_data = []
            
            for name, config in self.config["models"].items():
                status_data.append({
                    "model_name": name,
                    "active": config.get("active", True),
                    "last_retrained": config["last_retrained"],
                    "retraining_count": config["retraining_count"],
                    "retraining_attempts": config["retraining_attempts"],
                    "next_retraining": (
                        datetime.fromisoformat(config["last_retrained"]) + 
                        timedelta(days=config["retraining_interval_days"])
                    ).isoformat()
                })
            
            return pd.DataFrame(status_data)
    
    def manually_retrain(self, model_name: str):
        """
        Manually trigger retraining for a model.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.config["models"]:
            raise ValueError(f"Model '{model_name}' not found in scheduler")
        
        self.logger.info(f"Manually triggering retraining for model '{model_name}'")
        self._retrain_model(model_name)
