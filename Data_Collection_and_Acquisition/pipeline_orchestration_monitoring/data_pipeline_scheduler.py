# data_pipeline_scheduler.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
import os
import json
import schedule
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

class DataPipelineScheduler:
    """
    A class to define, schedule, and manage complex data workflow execution.
    Ensures all data processes run in correct order and without failure.
    """
    
    def __init__(self, config_dir="config/pipelines"):
        """
        Initialize Data Pipeline Scheduler
        
        Args:
            config_dir (str): Directory to store pipeline configurations
        """
        self.config_dir = config_dir
        self.logger = self._setup_logger()
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Pipeline definitions
        self.pipelines = {}
        self.pipeline_configs = {}
        self.pipeline_states = {}
        self.pipeline_dependencies = {}
        
        # Scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_listener(self._handle_job_event, 
                                    EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
        
        # Scheduler status
        self.is_running = False
        
        # Metrics
        self.scheduler_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'missed_executions': 0,
            'last_execution_time': None,
            'last_failed_execution': None,
            'last_missed_execution': None
        }
        
        # Load existing pipeline configurations
        self._load_pipeline_configs()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("DataPipelineScheduler")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("data_pipeline_scheduler.log")
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
    
    def _load_pipeline_configs(self):
        """Load pipeline configurations from files"""
        try:
            for filename in os.listdir(self.config_dir):
                if filename.endswith('.json'):
                    pipeline_id = filename[:-5]  # Remove .json extension
                    
                    with open(os.path.join(self.config_dir, filename), 'r') as f:
                        config = json.load(f)
                    
                    self.pipeline_configs[pipeline_id] = config
                    
                    # Initialize pipeline state
                    self.pipeline_states[pipeline_id] = {
                        'status': 'idle',
                        'last_execution': None,
                        'next_execution': None,
                        'last_success': None,
                        'last_failure': None,
                        'execution_count': 0,
                        'success_count': 0,
                        'failure_count': 0
                    }
                    
                    # Initialize dependencies
                    self.pipeline_dependencies[pipeline_id] = config.get('dependencies', [])
                    
                    self.logger.info(f"Loaded pipeline configuration for {pipeline_id}")
        except Exception as e:
            self.logger.error(f"Error loading pipeline configurations: {str(e)}")
    
    def _save_pipeline_config(self, pipeline_id):
        """Save pipeline configuration to file"""
        if pipeline_id not in self.pipeline_configs:
            return False
        
        try:
            config_file = os.path.join(self.config_dir, f"{pipeline_id}.json")
            
            with open(config_file, 'w') as f:
                json.dump(self.pipeline_configs[pipeline_id], f, indent=2)
            
            self.logger.debug(f"Saved pipeline configuration for {pipeline_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving pipeline configuration for {pipeline_id}: {str(e)}")
            return False
    
    def _handle_job_event(self, event):
        """Handle scheduler job events"""
        job_id = event.job_id
        
        if event.code == EVENT_JOB_EXECUTED:
            self.scheduler_metrics['total_executions'] += 1
            self.scheduler_metrics['successful_executions'] += 1
            self.scheduler_metrics['last_execution_time'] = datetime.now()
            
            # Update pipeline state
            if job_id in self.pipeline_states:
                self.pipeline_states[job_id]['last_execution'] = datetime.now()
                self.pipeline_states[job_id]['last_success'] = datetime.now()
                self.pipeline_states[job_id]['execution_count'] += 1
                self.pipeline_states[job_id]['success_count'] += 1
                self.pipeline_states[job_id]['status'] = 'idle'
            
            self.logger.debug(f"Pipeline {job_id} executed successfully")
            
        elif event.code == EVENT_JOB_ERROR:
            self.scheduler_metrics['total_executions'] += 1
            self.scheduler_metrics['failed_executions'] += 1
            self.scheduler_metrics['last_failed_execution'] = datetime.now()
            
            # Update pipeline state
            if job_id in self.pipeline_states:
                self.pipeline_states[job_id]['last_execution'] = datetime.now()
                self.pipeline_states[job_id]['last_failure'] = datetime.now()
                self.pipeline_states[job_id]['execution_count'] += 1
                self.pipeline_states[job_id]['failure_count'] += 1
                self.pipeline_states[job_id]['status'] = 'failed'
            
            self.logger.error(f"Pipeline {job_id} execution failed: {str(event.exception)}")
            
        elif event.code == EVENT_JOB_MISSED:
            self.scheduler_metrics['missed_executions'] += 1
            self.scheduler_metrics['last_missed_execution'] = datetime.now()
            
            # Update pipeline state
            if job_id in self.pipeline_states:
                self.pipeline_states[job_id]['status'] = 'missed'
            
            self.logger.warning(f"Pipeline {job_id} execution missed")
    
    def add_pipeline(self, pipeline_id, pipeline_config):
        """
        Add a pipeline
        
        Args:
            pipeline_id (str): Pipeline identifier
            pipeline_config (dict): Pipeline configuration
            
        Returns:
            bool: Whether successfully added pipeline
        """
        if pipeline_id in self.pipeline_configs:
            self.logger.warning(f"Pipeline {pipeline_id} already exists")
            return False
        
        # Validate configuration
        required_fields = ['name', 'description', 'schedule', 'steps']
        for field in required_fields:
            if field not in pipeline_config:
                self.logger.error(f"Missing required field in pipeline configuration: {field}")
                return False
        
        # Validate schedule
        schedule_config = pipeline_config['schedule']
        schedule_type = schedule_config.get('type')
        
        if schedule_type not in ['cron', 'interval']:
            self.logger.error(f"Invalid schedule type: {schedule_type}")
            return False
        
        if schedule_type == 'cron' and 'cron_expression' not in schedule_config:
            self.logger.error("Missing cron_expression in schedule configuration")
            return False
        
        if schedule_type == 'interval' and 'interval_seconds' not in schedule_config:
            self.logger.error("Missing interval_seconds in schedule configuration")
            return False
        
        # Add pipeline configuration
        self.pipeline_configs[pipeline_id] = pipeline_config
        
        # Initialize pipeline state
        self.pipeline_states[pipeline_id] = {
            'status': 'idle',
            'last_execution': None,
            'next_execution': None,
            'last_success': None,
            'last_failure': None,
            'execution_count': 0,
            'success_count': 0,
            'failure_count': 0
        }
        
        # Initialize dependencies
        self.pipeline_dependencies[pipeline_id] = pipeline_config.get('dependencies', [])
        
        # Save configuration
        self._save_pipeline_config(pipeline_id)
        
        # Schedule pipeline if scheduler is running
        if self.is_running:
            self._schedule_pipeline(pipeline_id)
        
        self.logger.info(f"Added pipeline {pipeline_id}")
        return True
    
    def remove_pipeline(self, pipeline_id):
        """
        Remove a pipeline
        
        Args:
            pipeline_id (str): Pipeline identifier
            
        Returns:
            bool: Whether successfully removed pipeline
        """
        if pipeline_id not in self.pipeline_configs:
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return False
        
        # Remove from scheduler if running
        if self.is_running and self.scheduler.get_job(pipeline_id):
            self.scheduler.remove_job(pipeline_id)
        
        # Remove pipeline configuration, state, and dependencies
        del self.pipeline_configs[pipeline_id]
        del self.pipeline_states[pipeline_id]
        del self.pipeline_dependencies[pipeline_id]
        
        # Remove configuration file
        config_file = os.path.join(self.config_dir, f"{pipeline_id}.json")
        if os.path.exists(config_file):
            os.remove(config_file)
        
        self.logger.info(f"Removed pipeline {pipeline_id}")
        return True
    
    def update_pipeline(self, pipeline_id, pipeline_config):
        """
        Update a pipeline
        
        Args:
            pipeline_id (str): Pipeline identifier
            pipeline_config (dict): Pipeline configuration
            
        Returns:
            bool: Whether successfully updated pipeline
        """
        if pipeline_id not in self.pipeline_configs:
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return False
        
        # Remove from scheduler if running
        if self.is_running and self.scheduler.get_job(pipeline_id):
            self.scheduler.remove_job(pipeline_id)
        
        # Update pipeline configuration
        self.pipeline_configs[pipeline_id] = pipeline_config
        
        # Update dependencies
        self.pipeline_dependencies[pipeline_id] = pipeline_config.get('dependencies', [])
        
        # Save configuration
        self._save_pipeline_config(pipeline_id)
        
        # Reschedule pipeline if scheduler is running
        if self.is_running:
            self._schedule_pipeline(pipeline_id)
        
        self.logger.info(f"Updated pipeline {pipeline_id}")
        return True
    
    def _schedule_pipeline(self, pipeline_id):
        """
        Schedule a pipeline
        
        Args:
            pipeline_id (str): Pipeline identifier
            
        Returns:
            bool: Whether successfully scheduled pipeline
        """
        if pipeline_id not in self.pipeline_configs:
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return False
        
        # Get pipeline configuration
        config = self.pipeline_configs[pipeline_id]
        schedule_config = config['schedule']
        schedule_type = schedule_config.get('type')
        
        # Create job function
        def execute_pipeline():
            self._execute_pipeline(pipeline_id)
        
        # Schedule based on type
        if schedule_type == 'cron':
            cron_expression = schedule_config.get('cron_expression')
            trigger = CronTrigger.from_crontab(cron_expression)
        elif schedule_type == 'interval':
            interval_seconds = schedule_config.get('interval_seconds')
            trigger = IntervalTrigger(seconds=interval_seconds)
        else:
            self.logger.error(f"Invalid schedule type: {schedule_type}")
            return False
        
        # Add job to scheduler
        self.scheduler.add_job(
            execute_pipeline,
            trigger=trigger,
            id=pipeline_id,
            name=config['name'],
            replace_existing=True
        )
        
        # Update next execution time
        job = self.scheduler.get_job(pipeline_id)
        if job:
            self.pipeline_states[pipeline_id]['next_execution'] = job.next_run_time
        
        self.logger.info(f"Scheduled pipeline {pipeline_id}")
        return True
    
    def _execute_pipeline(self, pipeline_id):
        """
        Execute a pipeline
        
        Args:
            pipeline_id (str): Pipeline identifier
        """
        if pipeline_id not in self.pipeline_configs:
            self.logger.error(f"Pipeline {pipeline_id} not found")
            return
        
        # Update pipeline state
        self.pipeline_states[pipeline_id]['status'] = 'running'
        
        try:
            # Get pipeline configuration
            config = self.pipeline_configs[pipeline_id]
            
            # Check dependencies
            dependencies = self.pipeline_dependencies.get(pipeline_id, [])
            for dep_id in dependencies:
                if dep_id in self.pipeline_states:
                    dep_state = self.pipeline_states[dep_id]
                    if dep_state['status'] == 'running':
                        self.logger.warning(f"Pipeline {pipeline_id} waiting for dependency {dep_id}")
                        time.sleep(1)  # Wait a bit and check again
                        if self.pipeline_states[dep_id]['status'] == 'running':
                            self.logger.error(f"Pipeline {pipeline_id} cannot start due to running dependency {dep_id}")
                            return
                    elif dep_state['last_execution'] is None:
                        self.logger.warning(f"Pipeline {pipeline_id} waiting for dependency {dep_id} to execute first")
                        return
                    elif dep_state['last_failure'] is not None and dep_state['last_failure'] > dep_state['last_success']:
                        self.logger.error(f"Pipeline {pipeline_id} cannot start due to failed dependency {dep_id}")
                        return
            
            # Execute steps
            steps = config['steps']
            for i, step in enumerate(steps):
                step_name = step.get('name', f"Step {i+1}")
                step_type = step.get('type')
                
                self.logger.info(f"Executing {step_name} of pipeline {pipeline_id}")
                
                # Execute based on type
                if step_type == 'function':
                    # Execute function
                    function_name = step.get('function')
                    if not hasattr(self, function_name):
                        self.logger.error(f"Function {function_name} not found")
                        raise Exception(f"Function {function_name} not found")
                    
                    function = getattr(self, function_name)
                    function(**step.get('parameters', {}))
                
                elif step_type == 'command':
                    # Execute command
                    command = step.get('command')
                    if not command:
                        self.logger.error("Missing command in step configuration")
                        raise Exception("Missing command in step configuration")
                    
                    # Execute command (placeholder)
                    self.logger.info(f"Executing command: {command}")
                    # In a real implementation, this would execute the command
                
                elif step_type == 'delay':
                    # Delay execution
                    delay_seconds = step.get('seconds', 1)
                    self.logger.info(f"Delaying execution for {delay_seconds} seconds")
                    time.sleep(delay_seconds)
                
                else:
                    self.logger.error(f"Unknown step type: {step_type}")
                    raise Exception(f"Unknown step type: {step_type}")
            
            # Update pipeline state
            self.pipeline_states[pipeline_id]['status'] = 'completed'
            
            self.logger.info(f"Pipeline {pipeline_id} executed successfully")
            
        except Exception as e:
            # Update pipeline state
            self.pipeline_states[pipeline_id]['status'] = 'failed'
            
            self.logger.error(f"Pipeline {pipeline_id} execution failed: {str(e)}")
            raise
    
    def start_scheduler(self):
        """
        Start the scheduler
        
        Returns:
            bool: Whether successfully started scheduler
        """
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return False
        
        # Schedule all pipelines
        for pipeline_id in self.pipeline_configs:
            self._schedule_pipeline(pipeline_id)
        
        # Start scheduler
        self.scheduler.start()
        self.is_running = True
        
        self.logger.info("Started pipeline scheduler")
        return True
    
    def stop_scheduler(self):
        """
        Stop the scheduler
        
        Returns:
            bool: Whether successfully stopped scheduler
        """
        if not self.is_running:
            self.logger.warning("Scheduler is not running")
            return False
        
        # Stop scheduler
        self.scheduler.shutdown(wait=True)
        self.is_running = False
        
        self.logger.info("Stopped pipeline scheduler")
        return True
    
    def execute_pipeline_now(self, pipeline_id):
        """
        Execute a pipeline immediately
        
        Args:
            pipeline_id (str): Pipeline identifier
            
        Returns:
            bool: Whether successfully executed pipeline
        """
        if pipeline_id not in self.pipeline_configs:
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return False
        
        try:
            # Execute pipeline
            self._execute_pipeline(pipeline_id)
            
            self.logger.info(f"Executed pipeline {pipeline_id} immediately")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing pipeline {pipeline_id} immediately: {str(e)}")
            return False
    
    def get_pipeline_config(self, pipeline_id):
        """
        Get pipeline configuration
        
        Args:
            pipeline_id (str): Pipeline identifier
            
        Returns:
            dict: Pipeline configuration or None if pipeline not found
        """
        if pipeline_id not in self.pipeline_configs:
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return None
        
        return self.pipeline_configs[pipeline_id].copy()
    
    def get_pipeline_state(self, pipeline_id):
        """
        Get pipeline state
        
        Args:
            pipeline_id (str): Pipeline identifier
            
        Returns:
            dict: Pipeline state or None if pipeline not found
        """
        if pipeline_id not in self.pipeline_states:
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return None
        
        return self.pipeline_states[pipeline_id].copy()
    
    def get_pipeline_dependencies(self, pipeline_id):
        """
        Get pipeline dependencies
        
        Args:
            pipeline_id (str): Pipeline identifier
            
        Returns:
            list: List of pipeline dependencies or None if pipeline not found
        """
        if pipeline_id not in self.pipeline_dependencies:
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return None
        
        return self.pipeline_dependencies[pipeline_id].copy()
    
    def get_all_pipeline_ids(self):
        """
        Get all pipeline IDs
        
        Returns:
            list: List of pipeline IDs
        """
        return list(self.pipeline_configs.keys())
    
    def get_scheduler_metrics(self):
        """
        Get scheduler metrics
        
        Returns:
            dict: Scheduler metrics
        """
        return self.scheduler_metrics.copy()
    
    def get_scheduled_jobs(self):
        """
        Get scheduled jobs
        
        Returns:
            list: List of scheduled jobs
        """
        jobs = []
        
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time,
                'trigger': str(job.trigger)
            })
        
        return jobs
    
    def check_dependencies(self, pipeline_id):
        """
        Check if pipeline dependencies are satisfied
        
        Args:
            pipeline_id (str): Pipeline identifier
            
        Returns:
            dict: Dependency check result
        """
        if pipeline_id not in self.pipeline_configs:
            self.logger.warning(f"Pipeline {pipeline_id} not found")
            return None
        
        result = {
            'pipeline_id': pipeline_id,
            'dependencies': [],
            'satisfied': True,
            'issues': []
        }
        
        # Check each dependency
        dependencies = self.pipeline_dependencies.get(pipeline_id, [])
        for dep_id in dependencies:
            dep_info = {
                'dependency_id': dep_id,
                'satisfied': True,
                'issue': None
            }
            
            if dep_id in self.pipeline_states:
                dep_state = self.pipeline_states[dep_id]
                
                if dep_state['status'] == 'running':
                    dep_info['satisfied'] = False
                    dep_info['issue'] = 'Dependency is currently running'
                    result['satisfied'] = False
                    result['issues'].append(f"Dependency {dep_id} is currently running")
                
                elif dep_state['last_execution'] is None:
                    dep_info['satisfied'] = False
                    dep_info['issue'] = 'Dependency has not been executed yet'
                    result['satisfied'] = False
                    result['issues'].append(f"Dependency {dep_id} has not been executed yet")
                
                elif dep_state['last_failure'] is not None and dep_state['last_failure'] > dep_state['last_success']:
                    dep_info['satisfied'] = False
                    dep_info['issue'] = 'Dependency last execution failed'
                    result['satisfied'] = False
                    result['issues'].append(f"Dependency {dep_id} last execution failed")
            else:
                dep_info['satisfied'] = False
                dep_info['issue'] = 'Dependency not found'
                result['satisfied'] = False
                result['issues'].append(f"Dependency {dep_id} not found")
            
            result['dependencies'].append(dep_info)
        
        return result
