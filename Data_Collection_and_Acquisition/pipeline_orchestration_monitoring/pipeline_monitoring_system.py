# pipeline_monitoring_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict, deque
import psutil
import requests

class PipelineMonitoringSystem:
    """
    A class to provide real-time monitoring and workflow dashboards for all data processes.
    Tracks performance metrics, error rates, and system health indicators.
    """
    
    def __init__(self, config_dir="config/monitoring", history_size=1000):
        """
        Initialize Pipeline Monitoring System
        
        Args:
            config_dir (str): Directory to store monitoring configurations
            history_size (int): Size of history to keep for each metric
        """
        self.config_dir = config_dir
        self.history_size = history_size
        self.logger = self._setup_logger()
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Monitoring configuration
        self.monitoring_config = self._load_monitoring_config()
        
        # Metrics history
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_size))
        self.alert_history = deque(maxlen=history_size)
        
        # Monitoring status
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Metrics
        self.current_metrics = {}
        
        # Alert configuration
        self.alert_config = self._load_alert_config()
        self.alert_handlers = {}
        
        # Dashboard configuration
        self.dashboard_config = self._load_dashboard_config()
        
        # Pipeline status tracking
        self.pipeline_status = {}
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("PipelineMonitoringSystem")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("pipeline_monitoring_system.log")
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
    
    def _load_monitoring_config(self):
        """Load monitoring configuration"""
        config_file = os.path.join(self.config_dir, "monitoring_config.json")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading monitoring configuration: {str(e)}")
        
        # Return default configuration
        return {
            "refresh_interval": 5,  # seconds
            "metrics": [
                {
                    "name": "cpu_usage",
                    "type": "system",
                    "unit": "%"
                },
                {
                    "name": "memory_usage",
                    "type": "system",
                    "unit": "%"
                },
                {
                    "name": "disk_usage",
                    "type": "system",
                    "unit": "%"
                },
                {
                    "name": "network_io",
                    "type": "system",
                    "unit": "bytes/s"
                },
                {
                    "name": "pipeline_execution_time",
                    "type": "pipeline",
                    "unit": "seconds"
                },
                {
                    "name": "pipeline_success_rate",
                    "type": "pipeline",
                    "unit": "%"
                },
                {
                    "name": "data_throughput",
                    "type": "pipeline",
                    "unit": "records/s"
                },
                {
                    "name": "error_rate",
                    "type": "pipeline",
                    "unit": "%"
                }
            ]
        }
    
    def _load_alert_config(self):
        """Load alert configuration"""
        config_file = os.path.join(self.config_dir, "alert_config.json")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading alert configuration: {str(e)}")
        
        # Return default configuration
        return {
            "enabled": True,
            "channels": ["log", "email"],
            "email": {
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "user@example.com",
                "password": "password",
                "from_addr": "monitoring@example.com",
                "to_addrs": ["admin@example.com"]
            },
            "rules": [
                {
                    "metric": "cpu_usage",
                    "condition": ">",
                    "threshold": 90,
                    "duration": 300,  # seconds
                    "severity": "high"
                },
                {
                    "metric": "memory_usage",
                    "condition": ">",
                    "threshold": 90,
                    "duration": 300,
                    "severity": "high"
                },
                {
                    "metric": "disk_usage",
                    "condition": ">",
                    "threshold": 90,
                    "duration": 300,
                    "severity": "high"
                },
                {
                    "metric": "pipeline_success_rate",
                    "condition": "<",
                    "threshold": 95,
                    "duration": 600,
                    "severity": "medium"
                },
                {
                    "metric": "error_rate",
                    "condition": ">",
                    "threshold": 5,
                    "duration": 300,
                    "severity": "medium"
                }
            ]
        }
    
    def _load_dashboard_config(self):
        """Load dashboard configuration"""
        config_file = os.path.join(self.config_dir, "dashboard_config.json")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading dashboard configuration: {str(e)}")
        
        # Return default configuration
        return {
            "layout": "grid",
            "panels": [
                {
                    "title": "System Metrics",
                    "type": "metrics",
                    "metrics": ["cpu_usage", "memory_usage", "disk_usage"],
                    "size": "medium"
                },
                {
                    "title": "Pipeline Metrics",
                    "type": "metrics",
                    "metrics": ["pipeline_execution_time", "pipeline_success_rate", "data_throughput", "error_rate"],
                    "size": "medium"
                },
                {
                    "title": "Recent Alerts",
                    "type": "alerts",
                    "limit": 10,
                    "size": "small"
                },
                {
                    "title": "Pipeline Status",
                    "type": "status",
                    "size": "small"
                }
            ]
        }
    
    def start_monitoring(self):
        """
        Start monitoring
        
        Returns:
            bool: Whether successfully started monitoring
        """
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return False
        
        # Start monitoring thread
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Started pipeline monitoring")
        return True
    
    def stop_monitoring(self):
        """
        Stop monitoring
        
        Returns:
            bool: Whether successfully stopped monitoring
        """
        if not self.is_monitoring:
            self.logger.warning("Monitoring is not running")
            return False
        
        # Stop monitoring thread
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped pipeline monitoring")
        return True
    
    def _monitor_system(self):
        """
        Internal method to monitor system, runs in separate thread
        """
        while self.is_monitoring:
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Check alerts
                self._check_alerts()
                
                # Sleep for refresh interval
                time.sleep(self.monitoring_config.get('refresh_interval', 5))
                
            except Exception as e:
                self.logger.error(f"Error monitoring system: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _collect_metrics(self):
        """Collect system and pipeline metrics"""
        timestamp = datetime.now()
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics()
        
        # Collect pipeline metrics
        pipeline_metrics = self._collect_pipeline_metrics()
        
        # Combine metrics
        metrics = {
            'timestamp': timestamp,
            'system': system_metrics,
            'pipeline': pipeline_metrics
        }
        
        # Update current metrics
        self.current_metrics = metrics
        
        # Add to history
        self.metrics_history['all'].append(metrics)
        
        # Add to individual metric histories
        for metric_name, metric_value in system_metrics.items():
            self.metrics_history[f"system.{metric_name}"].append({
                'timestamp': timestamp,
                'value': metric_value
            })
        
        for metric_name, metric_value in pipeline_metrics.items():
            self.metrics_history[f"pipeline.{metric_name}"].append({
                'timestamp': timestamp,
                'value': metric_value
            })
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Get network I/O
            net_io = psutil.net_io_counters()
            network_io = net_io.bytes_sent + net_io.bytes_recv
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'network_io': network_io
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0,
                'network_io': 0
            }
    
    def _collect_pipeline_metrics(self):
        """Collect pipeline metrics"""
        # Placeholder for pipeline metrics collection
        # In a real implementation, this would collect actual pipeline metrics
        
        return {
            'pipeline_execution_time': np.random.uniform(1, 60),
            'pipeline_success_rate': np.random.uniform(90, 100),
            'data_throughput': np.random.uniform(100, 10000),
            'error_rate': np.random.uniform(0, 10)
        }
    
    def _check_alerts(self):
        """Check alert conditions"""
        if not self.alert_config.get('enabled', True):
            return
        
        # Get current metrics
        system_metrics = self.current_metrics.get('system', {})
        pipeline_metrics = self.current_metrics.get('pipeline', {})
        
        # Check each alert rule
        for rule in self.alert_config.get('rules', []):
            metric = rule.get('metric')
            condition = rule.get('condition')
            threshold = rule.get('threshold')
            duration = rule.get('duration', 0)
            severity = rule.get('severity', 'medium')
            
            # Get metric value
            if metric.startswith('system.'):
                metric_value = system_metrics.get(metric[7:])  # Remove 'system.' prefix
            elif metric.startswith('pipeline.'):
                metric_value = pipeline_metrics.get(metric[9:])  # Remove 'pipeline.' prefix
            else:
                self.logger.warning(f"Unknown metric prefix for {metric}")
                continue
            
            if metric_value is None:
                continue
            
            # Check condition
            triggered = False
            if condition == '>' and metric_value > threshold:
                triggered = True
            elif condition == '<' and metric_value < threshold:
                triggered = True
            elif condition == '==' and metric_value == threshold:
                triggered = True
            elif condition == '!=' and metric_value != threshold:
                triggered = True
            elif condition == '>=' and metric_value >= threshold:
                triggered = True
            elif condition == '<=' and metric_value <= threshold:
                triggered = True
            
            if triggered:
                # Check if condition has been met for the required duration
                if duration > 0:
                    # Get metric history
                    metric_history = self.metrics_history.get(metric, [])
                    
                    # Check if condition has been met for the required duration
                    now = datetime.now()
                    duration_threshold = now - timedelta(seconds=duration)
                    
                    # Count how many times the condition has been met in the duration
                    count = 0
                    for record in metric_history:
                        if record['timestamp'] < duration_threshold:
                            break
                        
                        value = record['value']
                        
                        if condition == '>' and value > threshold:
                            count += 1
                        elif condition == '<' and value < threshold:
                            count += 1
                        elif condition == '==' and value == threshold:
                            count += 1
                        elif condition == '!=' and value != threshold:
                            count += 1
                        elif condition == '>=' and value >= threshold:
                            count += 1
                        elif condition == '<=' and value <= threshold:
                            count += 1
                    
                    # Check if condition has been met for enough time
                    if count < duration / self.monitoring_config.get('refresh_interval', 5):
                        continue
                
                # Create alert
                alert = {
                    'timestamp': datetime.now(),
                    'metric': metric,
                    'condition': condition,
                    'threshold': threshold,
                    'value': metric_value,
                    'severity': severity,
                    'message': f"Alert: {metric} {condition} {threshold} (current: {metric_value})"
                }
                
                # Add to alert history
                self.alert_history.append(alert)
                
                # Handle alert
                self._handle_alert(alert)
    
    def _handle_alert(self, alert):
        """Handle alert"""
        # Log alert
        severity = alert.get('severity', 'medium')
        message = alert.get('message', 'Unknown alert')
        
        if severity == 'high':
            self.logger.error(message)
        elif severity == 'medium':
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # Send alert through configured channels
        channels = self.alert_config.get('channels', ['log'])
        
        if 'email' in channels:
            self._send_email_alert(alert)
        
        if 'webhook' in channels:
            self._send_webhook_alert(alert)
        
        # Call custom alert handlers
        for handler_id, handler in self.alert_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler {handler_id}: {str(e)}")
    
    def _send_email_alert(self, alert):
        """Send alert via email"""
        try:
            # Get email configuration
            email_config = self.alert_config.get('email', {})
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port', 587)
            username = email_config.get('username')
            password = email_config.get('password')
            from_addr = email_config.get('from_addr')
            to_addrs = email_config.get('to_addrs', [])
            
            if not all([smtp_server, username, password, from_addr, to_addrs]):
                self.logger.warning("Incomplete email configuration, skipping email alert")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_addr
            msg['To'] = ', '.join(to_addrs)
            msg['Subject'] = f"Pipeline Alert: {alert.get('severity', 'medium').upper()}"
            
            # Add body
            body = f"""
            Alert Details:
            - Metric: {alert.get('metric')}
            - Condition: {alert.get('condition')} {alert.get('threshold')}
            - Current Value: {alert.get('value')}
            - Severity: {alert.get('severity')}
            - Timestamp: {alert.get('timestamp')}
            
            Message: {alert.get('message')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            self.logger.info(f"Sent email alert for {alert.get('metric')}")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
    
    def _send_webhook_alert(self, alert):
        """Send alert via webhook"""
        try:
            # Get webhook URL from alert configuration
            webhook_url = self.alert_config.get('webhook_url')
            
            if not webhook_url:
                self.logger.warning("Webhook URL not configured, skipping webhook alert")
                return
            
            # Prepare payload
            payload = {
                'timestamp': alert.get('timestamp').isoformat(),
                'metric': alert.get('metric'),
                'condition': alert.get('condition'),
                'threshold': alert.get('threshold'),
                'value': alert.get('value'),
                'severity': alert.get('severity'),
                'message': alert.get('message')
            }
            
            # Send webhook
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Sent webhook alert for {alert.get('metric')}")
            else:
                self.logger.error(f"Failed to send webhook alert: {response.status_code} {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {str(e)}")
    
    def add_alert_handler(self, handler_id, handler):
        """
        Add custom alert handler
        
        Args:
            handler_id (str): Handler identifier
            handler (function): Handler function
            
        Returns:
            bool: Whether successfully added handler
        """
        if handler_id in self.alert_handlers:
            self.logger.warning(f"Alert handler {handler_id} already exists")
            return False
        
        self.alert_handlers[handler_id] = handler
        self.logger.info(f"Added alert handler {handler_id}")
        return True
    
    def remove_alert_handler(self, handler_id):
        """
        Remove custom alert handler
        
        Args:
            handler_id (str): Handler identifier
            
        Returns:
            bool: Whether successfully removed handler
        """
        if handler_id not in self.alert_handlers:
            self.logger.warning(f"Alert handler {handler_id} not found")
            return False
        
        del self.alert_handlers[handler_id]
        self.logger.info(f"Removed alert handler {handler_id}")
        return True
    
    def get_current_metrics(self):
        """
        Get current metrics
        
        Returns:
            dict: Current metrics
        """
        return self.current_metrics.copy()
    
    def get_metrics_history(self, metric_name, count=None):
        """
        Get metrics history
        
        Args:
            metric_name (str): Metric name
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of metric records
        """
        if metric_name not in self.metrics_history:
            self.logger.warning(f"Metrics history for {metric_name} not found")
            return []
        
        if count is None:
            return list(self.metrics_history[metric_name])
        else:
            return list(self.metrics_history[metric_name])[-count:]
    
    def get_metrics_history_dataframe(self, metric_name, count=None):
        """
        Get metrics history as DataFrame
        
        Args:
            metric_name (str): Metric name
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing metrics history
        """
        history = self.get_metrics_history(metric_name, count)
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        
        # Convert timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_alert_history(self, count=None):
        """
        Get alert history
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of alert records
        """
        if count is None:
            return list(self.alert_history)
        else:
            return list(self.alert_history)[-count:]
    
    def get_alert_history_dataframe(self, count=None):
        """
        Get alert history as DataFrame
        
        Args:
            count (int): Number of records to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing alert history
        """
        history = self.get_alert_history(count)
        
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        
        # Convert timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def update_pipeline_status(self, pipeline_id, status, details=None):
        """
        Update pipeline status
        
        Args:
            pipeline_id (str): Pipeline identifier
            status (str): Pipeline status
            details (dict): Additional status details
            
        Returns:
            bool: Whether successfully updated status
        """
        self.pipeline_status[pipeline_id] = {
            'timestamp': datetime.now(),
            'status': status,
            'details': details or {}
        }
        
        self.logger.info(f"Updated pipeline status for {pipeline_id}: {status}")
        return True
    
    def get_pipeline_status(self, pipeline_id=None):
        """
        Get pipeline status
        
        Args:
            pipeline_id (str): Pipeline identifier, None means get all
            
        Returns:
            dict or list: Pipeline status or list of pipeline statuses
        """
        if pipeline_id is not None:
            return self.pipeline_status.get(pipeline_id)
        else:
            return self.pipeline_status.copy()
    
    def get_dashboard_config(self):
        """
        Get dashboard configuration
        
        Returns:
            dict: Dashboard configuration
        """
        return self.dashboard_config.copy()
    
    def update_dashboard_config(self, config):
        """
        Update dashboard configuration
        
        Args:
            config (dict): Dashboard configuration
            
        Returns:
            bool: Whether successfully updated configuration
        """
        try:
            # Validate configuration
            required_fields = ['layout', 'panels']
            for field in required_fields:
                if field not in config:
                    self.logger.error(f"Missing required field in dashboard configuration: {field}")
                    return False
            
            # Update configuration
            self.dashboard_config = config
            
            # Save configuration
            config_file = os.path.join(self.config_dir, "dashboard_config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info("Updated dashboard configuration")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard configuration: {str(e)}")
            return False
    
    def generate_dashboard_data(self):
        """
        Generate data for dashboard
        
        Returns:
            dict: Dashboard data
        """
        dashboard_data = {
            'timestamp': datetime.now(),
            'metrics': {},
            'alerts': self.get_alert_history(10),
            'pipeline_status': self.get_pipeline_status()
        }
        
        # Add metrics for each panel
        for panel in self.dashboard_config.get('panels', []):
            if panel.get('type') == 'metrics':
                panel_metrics = panel.get('metrics', [])
                panel_title = panel.get('title', 'Metrics')
                
                dashboard_data['metrics'][panel_title] = {}
                
                for metric in panel_metrics:
                    # Get current value
                    if metric.startswith('system.'):
                        current_value = self.current_metrics.get('system', {}).get(metric[7:])
                    elif metric.startswith('pipeline.'):
                        current_value = self.current_metrics.get('pipeline', {}).get(metric[9:])
                    else:
                        current_value = None
                    
                    # Get history
                    history = self.get_metrics_history(metric, 100)
                    
                    dashboard_data['metrics'][panel_title][metric] = {
                        'current': current_value,
                        'history': history
                    }
        
        return dashboard_data
    
    def export_metrics_to_csv(self, filename, metric_names=None, start_time=None, end_time=None):
        """
        Export metrics to CSV file
        
        Args:
            filename (str): Filename
            metric_names (list): List of metric names to export, None means export all
            start_time (datetime): Start time for export
            end_time (datetime): End time for export
            
        Returns:
            bool: Whether successfully exported
        """
        try:
            # Get all metrics if not specified
            if metric_names is None:
                metric_names = [m['name'] for m in self.monitoring_config.get('metrics', [])]
                metric_names = [f"system.{m}" for m in metric_names] + [f"pipeline.{m}" for m in metric_names]
            
            # Create DataFrame
            dfs = []
            
            for metric_name in metric_names:
                df = self.get_metrics_history_dataframe(metric_name)
                
                if df.empty:
                    continue
                
                # Add metric name column
                df['metric'] = metric_name
                
                # Filter by time range if specified
                if start_time is not None:
                    df = df[df['timestamp'] >= start_time]
                
                if end_time is not None:
                    df = df[df['timestamp'] <= end_time]
                
                dfs.append(df)
            
            if not dfs:
                self.logger.warning("No metrics data to export")
                return False
            
            # Combine DataFrames
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Export to CSV
            combined_df.to_csv(filename, index=False)
            
            self.logger.info(f"Exported metrics to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics to CSV: {str(e)}")
            return False
    
    def export_alerts_to_csv(self, filename, start_time=None, end_time=None):
        """
        Export alerts to CSV file
        
        Args:
            filename (str): Filename
            start_time (datetime): Start time for export
            end_time (datetime): End time for export
            
        Returns:
            bool: Whether successfully exported
        """
        try:
            # Get alerts
            df = self.get_alert_history_dataframe()
            
            if df.empty:
                self.logger.warning("No alerts data to export")
                return False
            
            # Filter by time range if specified
            if start_time is not None:
                df = df[df['timestamp'] >= start_time]
            
            if end_time is not None:
                df = df[df['timestamp'] <= end_time]
            
            # Export to CSV
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Exported alerts to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting alerts to CSV: {str(e)}")
            return False
