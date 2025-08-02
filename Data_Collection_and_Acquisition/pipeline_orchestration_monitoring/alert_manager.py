# alert_manager.py
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
import requests

class AlertManager:
    """
    A class to send alerts and notifications to the team based on configurable thresholds.
    Supports multiple notification channels and escalation policies.
    """
    
    def __init__(self, config_dir="config/alerts"):
        """
        Initialize Alert Manager
        
        Args:
            config_dir (str): Directory to store alert configurations
        """
        self.config_dir = config_dir
        self.logger = self._setup_logger()
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Alert configuration
        self.alert_config = self._load_alert_config()
        
        # Alert history
        self.alert_history = []
        self.alert_history_file = os.path.join(config_dir, "alert_history.json")
        self._load_alert_history()
        
        # Alert handlers
        self.alert_handlers = {}
        
        # Alert queues
        self.alert_queues = {}
        self.alert_threads = {}
        
        # Metrics
        self.alert_metrics = {
            'total_alerts': 0,
            'alerts_by_severity': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'alerts_by_channel': {},
            'last_alert_time': None
        }
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("AlertManager")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("alert_manager.log")
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
            "channels": {
                "log": {
                    "enabled": True
                },
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "username": "user@example.com",
                    "password": "password",
                    "from_addr": "alerts@example.com",
                    "to_addrs": ["admin@example.com"]
                },
                "webhook": {
                    "enabled": True,
                    "url": "https://example.com/webhook"
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "https://hooks.slack.com/services/..."
                }
            },
            "rules": [
                {
                    "name": "High CPU Usage",
                    "condition": "system.cpu_usage > 90",
                    "duration": 300,
                    "severity": "high",
                    "channels": ["log", "email"],
                    "message": "High CPU usage detected: {value}%"
                },
                {
                    "name": "High Memory Usage",
                    "condition": "system.memory_usage > 90",
                    "duration": 300,
                    "severity": "high",
                    "channels": ["log", "email"],
                    "message": "High memory usage detected: {value}%"
                },
                {
                    "name": "Pipeline Failure",
                    "condition": "pipeline.status == 'failed'",
                    "duration": 0,
                    "severity": "critical",
                    "channels": ["log", "email", "webhook"],
                    "message": "Pipeline {pipeline_id} failed"
                },
                {
                    "name": "Data Quality Issue",
                    "condition": "data.quality_score < 80",
                    "duration": 0,
                    "severity": "medium",
                    "channels": ["log", "email"],
                    "message": "Data quality issue detected: score {value}"
                }
            ],
            "escalation_policy": {
                "enabled": True,
                "rules": [
                    {
                        "condition": "alert.severity == 'critical' and alert.escalation_level < 2",
                        "action": "escalate",
                        "escalation_level": 2,
                        "channels": ["log", "email", "webhook", "slack"]
                    },
                    {
                        "condition": "alert.severity == 'high' and alert.escalation_level < 1",
                        "action": "escalate",
                        "escalation_level": 1,
                        "channels": ["log", "email", "webhook"]
                    }
                ]
            }
        }
    
    def _load_alert_history(self):
        """Load alert history from file"""
        if os.path.exists(self.alert_history_file):
            try:
                with open(self.alert_history_file, 'r') as f:
                    self.alert_history = json.load(f)
                
                # Convert timestamp strings to datetime objects
                for alert in self.alert_history:
                    if 'timestamp' in alert and isinstance(alert['timestamp'], str):
                        alert['timestamp'] = datetime.fromisoformat(alert['timestamp'])
                
                self.logger.info(f"Loaded {len(self.alert_history)} alerts from history")
            except Exception as e:
                self.logger.error(f"Error loading alert history: {str(e)}")
                self.alert_history = []
    
    def _save_alert_history(self):
        """Save alert history to file"""
        try:
            # Create a copy of alert history with timestamp strings
            history_copy = []
            for alert in self.alert_history:
                alert_copy = alert.copy()
                if 'timestamp' in alert_copy and isinstance(alert_copy['timestamp'], datetime):
                    alert_copy['timestamp'] = alert_copy['timestamp'].isoformat()
                history_copy.append(alert_copy)
            
            # Save to file
            with open(self.alert_history_file, 'w') as f:
                json.dump(history_copy, f, indent=2)
            
            self.logger.debug("Saved alert history")
        except Exception as e:
            self.logger.error(f"Error saving alert history: {str(e)}")
    
    def add_alert_rule(self, rule):
        """
        Add alert rule
        
        Args:
            rule (dict): Alert rule
            
        Returns:
            bool: Whether successfully added rule
        """
        # Validate rule
        required_fields = ['name', 'condition', 'severity', 'channels']
        for field in required_fields:
            if field not in rule:
                self.logger.error(f"Missing required field in alert rule: {field}")
                return False
        
        # Add rule
        self.alert_config['rules'].append(rule)
        
        # Save configuration
        self._save_alert_config()
        
        self.logger.info(f"Added alert rule: {rule['name']}")
        return True
    
    def remove_alert_rule(self, rule_name):
        """
        Remove alert rule
        
        Args:
            rule_name (str): Rule name
            
        Returns:
            bool: Whether successfully removed rule
        """
        # Find and remove rule
        for i, rule in enumerate(self.alert_config['rules']):
            if rule.get('name') == rule_name:
                self.alert_config['rules'].pop(i)
                
                # Save configuration
                self._save_alert_config()
                
                self.logger.info(f"Removed alert rule: {rule_name}")
                return True
        
        self.logger.warning(f"Alert rule not found: {rule_name}")
        return False
    
    def update_alert_rule(self, rule_name, rule):
        """
        Update alert rule
        
        Args:
            rule_name (str): Rule name
            rule (dict): Alert rule
            
        Returns:
            bool: Whether successfully updated rule
        """
        # Find and update rule
        for i, existing_rule in enumerate(self.alert_config['rules']):
            if existing_rule.get('name') == rule_name:
                self.alert_config['rules'][i] = rule
                
                # Save configuration
                self._save_alert_config()
                
                self.logger.info(f"Updated alert rule: {rule_name}")
                return True
        
        self.logger.warning(f"Alert rule not found: {rule_name}")
        return False
    
    def _save_alert_config(self):
        """Save alert configuration to file"""
        try:
            config_file = os.path.join(self.config_dir, "alert_config.json")
            with open(config_file, 'w') as f:
                json.dump(self.alert_config, f, indent=2)
            
            self.logger.debug("Saved alert configuration")
        except Exception as e:
            self.logger.error(f"Error saving alert configuration: {str(e)}")
    
    def check_alerts(self, metrics):
        """
        Check alert conditions against metrics
        
        Args:
            metrics (dict): Current metrics
            
        Returns:
            list: List of triggered alerts
        """
        triggered_alerts = []
        
        # Check each rule
        for rule in self.alert_config.get('rules', []):
            # Parse condition
            condition = rule.get('condition')
            if not condition:
                continue
            
            # Evaluate condition
            try:
                # Create a local namespace for evaluation
                namespace = {'system': {}, 'pipeline': {}, 'data': {}, 'alert': {'escalation_level': 0}}
                
                # Add system metrics
                if 'system' in metrics:
                    namespace['system'].update(metrics['system'])
                
                # Add pipeline metrics
                if 'pipeline' in metrics:
                    namespace['pipeline'].update(metrics['pipeline'])
                
                # Add data metrics
                if 'data' in metrics:
                    namespace['data'].update(metrics['data'])
                
                # Evaluate condition
                triggered = eval(condition, {"__builtins__": {}}, namespace)
                
                if triggered:
                    # Create alert
                    alert = {
                        'timestamp': datetime.now(),
                        'rule_name': rule.get('name'),
                        'condition': condition,
                        'severity': rule.get('severity', 'medium'),
                        'channels': rule.get('channels', ['log']),
                        'message': rule.get('message', 'Alert triggered'),
                        'escalation_level': 0,
                        'metrics': metrics
                    }
                    
                    # Format message with metric values
                    message = alert['message']
                    for metric_type, metric_values in metrics.items():
                        for metric_name, metric_value in metric_values.items():
                            placeholder = f"{{{metric_type}.{metric_name}}}"
                            if placeholder in message:
                                message = message.replace(placeholder, str(metric_value))
                    
                    alert['message'] = message
                    
                    # Add to triggered alerts
                    triggered_alerts.append(alert)
                    
                    # Process alert
                    self._process_alert(alert)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating alert condition '{condition}': {str(e)}")
        
        return triggered_alerts
    
    def _process_alert(self, alert):
        """
        Process alert
        
        Args:
            alert (dict): Alert to process
        """
        # Add to history
        self.alert_history.append(alert)
        
        # Limit history size
        max_history = 1000
        if len(self.alert_history) > max_history:
            self.alert_history = self.alert_history[-max_history:]
        
        # Save history
        self._save_alert_history()
        
        # Update metrics
        self.alert_metrics['total_alerts'] += 1
        self.alert_metrics['alerts_by_severity'][alert.get('severity', 'medium')] += 1
        self.alert_metrics['last_alert_time'] = alert['timestamp']
        
        # Check escalation policy
        if self.alert_config.get('escalation_policy', {}).get('enabled', False):
            self._check_escalation(alert)
        
        # Send alert through channels
        channels = alert.get('channels', ['log'])
        
        for channel in channels:
            if channel == 'log':
                self._send_log_alert(alert)
                self._update_channel_metrics('log')
            elif channel == 'email':
                self._send_email_alert(alert)
                self._update_channel_metrics('email')
            elif channel == 'webhook':
                self._send_webhook_alert(alert)
                self._update_channel_metrics('webhook')
            elif channel == 'slack':
                self._send_slack_alert(alert)
                self._update_channel_metrics('slack')
        
        # Call custom alert handlers
        for handler_id, handler in self.alert_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler {handler_id}: {str(e)}")
    
    def _check_escalation(self, alert):
        """
        Check if alert should be escalated
        
        Args:
            alert (dict): Alert to check
        """
        escalation_policy = self.alert_config.get('escalation_policy', {})
        escalation_rules = escalation_policy.get('rules', [])
        
        for rule in escalation_rules:
            # Parse condition
            condition = rule.get('condition')
            if not condition:
                continue
            
            # Evaluate condition
            try:
                # Create a local namespace for evaluation
                namespace = {'alert': alert}
                
                # Evaluate condition
                should_escalate = eval(condition, {"__builtins__": {}}, namespace)
                
                if should_escalate:
                    # Get escalation action
                    action = rule.get('action')
                    
                    if action == 'escalate':
                        # Update escalation level
                        alert['escalation_level'] = rule.get('escalation_level', 1)
                        
                        # Update channels
                        alert['channels'] = rule.get('channels', alert['channels'])
                        
                        # Add escalation note to message
                        alert['message'] = f"[ESCALATED] {alert['message']}"
                        
                        self.logger.info(f"Escalated alert to level {alert['escalation_level']}: {alert['rule_name']}")
                        
            except Exception as e:
                self.logger.error(f"Error evaluating escalation condition '{condition}': {str(e)}")
    
    def _send_log_alert(self, alert):
        """
        Send alert via log
        
        Args:
            alert (dict): Alert to send
        """
        severity = alert.get('severity', 'medium')
        message = alert.get('message', 'Unknown alert')
        
        if severity == 'critical':
            self.logger.critical(message)
        elif severity == 'high':
            self.logger.error(message)
        elif severity == 'medium':
            self.logger.warning(message)
        elif severity == 'low':
            self.logger.info(message)
        else:
            self.logger.info(message)
    
    def _send_email_alert(self, alert):
        """
        Send alert via email
        
        Args:
            alert (dict): Alert to send
        """
        try:
            # Get email configuration
            email_config = self.alert_config.get('channels', {}).get('email', {})
            
            if not email_config.get('enabled', False):
                return
            
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
            
            # Set subject based on severity
            severity = alert.get('severity', 'medium')
            if severity == 'critical':
                msg['Subject'] = f"[CRITICAL] {alert.get('rule_name', 'Alert')}"
            elif severity == 'high':
                msg['Subject'] = f"[HIGH] {alert.get('rule_name', 'Alert')}"
            elif severity == 'medium':
                msg['Subject'] = f"[MEDIUM] {alert.get('rule_name', 'Alert')}"
            else:
                msg['Subject'] = f"[LOW] {alert.get('rule_name', 'Alert')}"
            
            # Add body
            body = f"""
            Alert Details:
            - Rule: {alert.get('rule_name')}
            - Severity: {alert.get('severity')}
            - Timestamp: {alert.get('timestamp')}
            
            Message: {alert.get('message')}
            
            Metrics:
            """
            
            # Add metrics to body
            metrics = alert.get('metrics', {})
            for metric_type, metric_values in metrics.items():
                body += f"- {metric_type}:\n"
                for metric_name, metric_value in metric_values.items():
                    body += f"  - {metric_name}: {metric_value}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            self.logger.info(f"Sent email alert for {alert.get('rule_name')}")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
    
    def _send_webhook_alert(self, alert):
        """
        Send alert via webhook
        
        Args:
            alert (dict): Alert to send
        """
        try:
            # Get webhook configuration
            webhook_config = self.alert_config.get('channels', {}).get('webhook', {})
            
            if not webhook_config.get('enabled', False):
                return
            
            webhook_url = webhook_config.get('url')
            
            if not webhook_url:
                self.logger.warning("Webhook URL not configured, skipping webhook alert")
                return
            
            # Prepare payload
            payload = {
                'timestamp': alert.get('timestamp').isoformat(),
                'rule_name': alert.get('rule_name'),
                'severity': alert.get('severity'),
                'message': alert.get('message'),
                'metrics': alert.get('metrics', {}),
                'escalation_level': alert.get('escalation_level', 0)
            }
            
            # Send webhook
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Sent webhook alert for {alert.get('rule_name')}")
            else:
                self.logger.error(f"Failed to send webhook alert: {response.status_code} {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {str(e)}")
    
    def _send_slack_alert(self, alert):
        """
        Send alert via Slack webhook
        
        Args:
            alert (dict): Alert to send
        """
        try:
            # Get Slack configuration
            slack_config = self.alert_config.get('channels', {}).get('slack', {})
            
            if not slack_config.get('enabled', False):
                return
            
            webhook_url = slack_config.get('webhook_url')
            
            if not webhook_url:
                self.logger.warning("Slack webhook URL not configured, skipping Slack alert")
                return
            
            # Prepare payload
            # Determine color based on severity
            severity = alert.get('severity', 'medium')
            if severity == 'critical':
                color = '#ff0000'  # Red
            elif severity == 'high':
                color = '#ff9900'  # Orange
            elif severity == 'medium':
                color = '#ffcc00'  # Yellow
            else:
                color = '#36a64f'  # Green
            
            # Create message
            payload = {
                'attachments': [
                    {
                        'color': color,
                        'title': f"{alert.get('severity', 'medium').upper()} Alert: {alert.get('rule_name')}",
                        'text': alert.get('message'),
                        'ts': alert.get('timestamp', datetime.now()).timestamp(),
                        'fields': [
                            {
                                'title': 'Severity',
                                'value': alert.get('severity', 'medium'),
                                'short': True
                            },
                            {
                                'title': 'Rule',
                                'value': alert.get('rule_name'),
                                'short': True
                            },
                            {
                                'title': 'Timestamp',
                                'value': alert.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                                'short': True
                            }
                        ]
                    }
                ]
            }
            
            # Send webhook
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Sent Slack alert for {alert.get('rule_name')}")
            else:
                self.logger.error(f"Failed to send Slack alert: {response.status_code} {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {str(e)}")
    
    def _update_channel_metrics(self, channel):
        """
        Update channel metrics
        
        Args:
            channel (str): Channel name
        """
        if channel not in self.alert_metrics['alerts_by_channel']:
            self.alert_metrics['alerts_by_channel'][channel] = 0
        
        self.alert_metrics['alerts_by_channel'][channel] += 1
    
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
    
    def get_alert_history(self, count=None, severity=None, rule_name=None, start_time=None, end_time=None):
        """
        Get alert history
        
        Args:
            count (int): Number of alerts to get, None means get all
            severity (str): Filter by severity
            rule_name (str): Filter by rule name
            start_time (datetime): Start time for filter
            end_time (datetime): End time for filter
            
        Returns:
            list: List of alerts
        """
        # Filter alerts
        filtered_alerts = []
        
        for alert in self.alert_history:
            # Filter by severity
            if severity is not None and alert.get('severity') != severity:
                continue
            
            # Filter by rule name
            if rule_name is not None and alert.get('rule_name') != rule_name:
                continue
            
            # Filter by start time
            if start_time is not None and alert.get('timestamp') < start_time:
                continue
            
            # Filter by end time
            if end_time is not None and alert.get('timestamp') > end_time:
                continue
            
            filtered_alerts.append(alert)
        
        # Limit count if specified
        if count is not None:
            filtered_alerts = filtered_alerts[-count:]
        
        return filtered_alerts
    
    def get_alert_history_dataframe(self, count=None, severity=None, rule_name=None, start_time=None, end_time=None):
        """
        Get alert history as DataFrame
        
        Args:
            count (int): Number of alerts to get, None means get all
            severity (str): Filter by severity
            rule_name (str): Filter by rule name
            start_time (datetime): Start time for filter
            end_time (datetime): End time for filter
            
        Returns:
            pandas.DataFrame: DataFrame containing alert history
        """
        alerts = self.get_alert_history(count, severity, rule_name, start_time, end_time)
        
        if not alerts:
            return pd.DataFrame()
        
        df = pd.DataFrame(alerts)
        
        # Convert timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_alert_metrics(self):
        """
        Get alert metrics
        
        Returns:
            dict: Alert metrics
        """
        return self.alert_metrics.copy()
    
    def get_alert_rules(self):
        """
        Get alert rules
        
        Returns:
            list: List of alert rules
        """
        return self.alert_config.get('rules', []).copy()
    
    def get_alert_rule(self, rule_name):
        """
        Get alert rule
        
        Args:
            rule_name (str): Rule name
            
        Returns:
            dict: Alert rule or None if not found
        """
        for rule in self.alert_config.get('rules', []):
            if rule.get('name') == rule_name:
                return rule.copy()
        
        return None
    
    def test_alert_rule(self, rule_name, metrics):
        """
        Test alert rule
        
        Args:
            rule_name (str): Rule name
            metrics (dict): Test metrics
            
        Returns:
            bool: Whether rule would trigger
        """
        rule = self.get_alert_rule(rule_name)
        if not rule:
            self.logger.warning(f"Alert rule not found: {rule_name}")
            return False
        
        # Parse condition
        condition = rule.get('condition')
        if not condition:
            return False
        
        # Evaluate condition
        try:
            # Create a local namespace for evaluation
            namespace = {'system': {}, 'pipeline': {}, 'data': {}, 'alert': {'escalation_level': 0}}
            
            # Add system metrics
            if 'system' in metrics:
                namespace['system'].update(metrics['system'])
            
            # Add pipeline metrics
            if 'pipeline' in metrics:
                namespace['pipeline'].update(metrics['pipeline'])
            
            # Add data metrics
            if 'data' in metrics:
                namespace['data'].update(metrics['data'])
            
            # Evaluate condition
            return eval(condition, {"__builtins__": {}}, namespace)
            
        except Exception as e:
            self.logger.error(f"Error testing alert rule '{rule_name}': {str(e)}")
            return False
    
    def export_alerts_to_csv(self, filename, count=None, severity=None, rule_name=None, start_time=None, end_time=None):
        """
        Export alerts to CSV file
        
        Args:
            filename (str): Filename
            count (int): Number of alerts to export, None means export all
            severity (str): Filter by severity
            rule_name (str): Filter by rule name
            start_time (datetime): Start time for filter
            end_time (datetime): End time for filter
            
        Returns:
            bool: Whether successfully exported
        """
        try:
            df = self.get_alert_history_dataframe(count, severity, rule_name, start_time, end_time)
            
            if df.empty:
                self.logger.warning("No alerts to export")
                return False
            
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Exported {len(df)} alerts to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting alerts to CSV: {str(e)}")
            return False
    
    def clear_alert_history(self):
        """
        Clear alert history
        
        Returns:
            int: Number of alerts cleared
        """
        count = len(self.alert_history)
        self.alert_history = []
        self._save_alert_history()
        
        self.logger.info(f"Cleared {count} alerts from history")
        return count
