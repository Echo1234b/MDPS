# data_source_profiler.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataSourceProfiler:
    """
    A class to generate statistical summaries for each new data batch.
    Captures data characteristics, distributions, and quality metrics for analysis.
    """
    
    def __init__(self, profile_dir="profiles/data_sources"):
        """
        Initialize Data Source Profiler
        
        Args:
            profile_dir (str): Directory to store profiles
        """
        self.profile_dir = profile_dir
        self.logger = self._setup_logger()
        
        # Create profile directory if it doesn't exist
        os.makedirs(profile_dir, exist_ok=True)
        
        # Data sources
        self.data_sources = {}
        self.profiles = {}
        self.profile_history = {}
        
        # Profiling status
        self.is_profiling = False
        self.profiling_thread = None
        
        # Profile metrics
        self.profile_metrics = {
            'total_profiles': 0,
            'last_profile_time': None,
            'data_sources_count': 0
        }
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("DataSourceProfiler")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("data_source_profiler.log")
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
    
    def add_data_source(self, source_id, source_name=None, fields=None):
        """
        Add data source for profiling
        
        Args:
            source_id (str): Source identifier
            source_name (str): Source display name
            fields (list): List of fields to profile, None means profile all
            
        Returns:
            bool: Whether successfully added source
        """
        if source_id in self.data_sources:
            self.logger.warning(f"Data source {source_id} already exists")
            return False
        
        # Add source
        self.data_sources[source_id] = {
            'source_name': source_name or source_id,
            'fields': fields,
            'created_at': datetime.now()
        }
        
        # Initialize profile
        self.profiles[source_id] = {
            'source_id': source_id,
            'source_name': source_name or source_id,
            'fields': {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Initialize profile history
        self.profile_history[source_id] = []
        
        # Update metrics
        self.profile_metrics['data_sources_count'] += 1
        
        # If this is the first source, start the profiling thread
        if not self.is_profiling:
            self.is_profiling = True
            self.profiling_thread = threading.Thread(target=self._profile_data)
            self.profiling_thread.daemon = True
            self.profiling_thread.start()
        
        self.logger.info(f"Added data source {source_id} for profiling")
        return True
    
    def remove_data_source(self, source_id):
        """
        Remove data source from profiling
        
        Args:
            source_id (str): Source identifier
            
        Returns:
            bool: Whether successfully removed source
        """
        if source_id not in self.data_sources:
            self.logger.warning(f"Data source {source_id} not found")
            return False
        
        # Save profile before removing
        self._save_profile(source_id)
        
        # Remove source, profile, and history
        del self.data_sources[source_id]
        del self.profiles[source_id]
        del self.profile_history[source_id]
        
        # Update metrics
        self.profile_metrics['data_sources_count'] -= 1
        
        # If no more sources, stop the profiling thread
        if not self.data_sources and self.is_profiling:
            self.is_profiling = False
            if self.profiling_thread and self.profiling_thread.is_alive():
                self.profiling_thread.join(timeout=5)
        
        self.logger.info(f"Removed data source {source_id} from profiling")
        return True
    
    def profile_data(self, source_id, data):
        """
        Profile data
        
        Args:
            source_id (str): Source identifier
            data (dict or list): Data to profile
            
        Returns:
            bool: Whether successfully profiled data
        """
        if source_id not in self.data_sources:
            self.logger.warning(f"Data source {source_id} not found")
            return False
        
        # Convert to list if single record
        if isinstance(data, dict):
            data = [data]
        
        if not data:
            return False
        
        # Get fields to profile
        fields = self.data_sources[source_id]['fields']
        if fields is None:
            # Get all fields from first record
            if isinstance(data[0], dict):
                fields = list(data[0].keys())
            else:
                fields = ['value']
        
        # Profile each field
        for field in fields:
            # Extract field values
            values = []
            for record in data:
                if isinstance(record, dict):
                    if field in record:
                        value = record[field]
                        # Convert to numeric if possible
                        try:
                            value = float(value)
                            values.append(value)
                        except (ValueError, TypeError):
                            pass
                else:
                    # Treat as single value
                    try:
                        value = float(record)
                        values.append(value)
                    except (ValueError, TypeError):
                        pass
            
            # Skip if no numeric values
            if not values:
                continue
            
            # Calculate statistics
            field_profile = self._calculate_field_stats(values)
            
            # Update profile
            if field not in self.profiles[source_id]['fields']:
                self.profiles[source_id]['fields'][field] = {
                    'profile_count': 0,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'median': None,
                    'std': None,
                    'variance': None,
                    'skewness': None,
                    'kurtosis': None,
                    'percentiles': {},
                    'histogram': {},
                    'updated_at': datetime.now().isoformat()
                }
            
            # Update field profile
            field_profile_obj = self.profiles[source_id]['fields'][field]
            field_profile_obj['profile_count'] += 1
            
            # Update min/max
            if field_profile_obj['min'] is None or field_profile['min'] < field_profile_obj['min']:
                field_profile_obj['min'] = field_profile['min']
            
            if field_profile_obj['max'] is None or field_profile['max'] > field_profile_obj['max']:
                field_profile_obj['max'] = field_profile['max']
            
            # Update other statistics
            field_profile_obj['mean'] = field_profile['mean']
            field_profile_obj['median'] = field_profile['median']
            field_profile_obj['std'] = field_profile['std']
            field_profile_obj['variance'] = field_profile['variance']
            field_profile_obj['skewness'] = field_profile['skewness']
            field_profile_obj['kurtosis'] = field_profile['kurtosis']
            
            # Update percentiles
            for p, value in field_profile['percentiles'].items():
                field_profile_obj['percentiles'][p] = value
            
            # Update histogram
            for bin_key, count in field_profile['histogram'].items():
                if bin_key in field_profile_obj['histogram']:
                    field_profile_obj['histogram'][bin_key] += count
                else:
                    field_profile_obj['histogram'][bin_key] = count
            
            field_profile_obj['updated_at'] = datetime.now().isoformat()
        
        # Update profile timestamp
        self.profiles[source_id]['updated_at'] = datetime.now().isoformat()
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'record_count': len(data),
            'fields_profiled': len(fields)
        }
        self.profile_history[source_id].append(history_entry)
        
        # Limit history size
        if len(self.profile_history[source_id]) > 1000:
            self.profile_history[source_id] = self.profile_history[source_id][-1000:]
        
        # Update metrics
        self.profile_metrics['total_profiles'] += 1
        self.profile_metrics['last_profile_time'] = datetime.now()
        
        return True
    
    def _profile_data(self):
        """
        Internal method to profile data, runs in separate thread
        """
        while self.is_profiling:
            try:
                # Save profiles periodically
                for source_id in self.profiles:
                    self._save_profile(source_id)
                
                # Sleep for a while
                time.sleep(300)  # Save profiles every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error profiling data: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _calculate_field_stats(self, values):
        """
        Calculate statistics for field values
        
        Args:
            values (list): List of numeric values
            
        Returns:
            dict: Field statistics
        """
        # Convert to numpy array
        arr = np.array(values)
        
        # Calculate basic statistics
        stats = {
            'count': len(arr),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'variance': float(np.var(arr)),
            'skewness': float(stats.skew(arr)),
            'kurtosis': float(stats.kurtosis(arr)),
            'percentiles': {},
            'histogram': {}
        }
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats['percentiles'][p] = float(np.percentile(arr, p))
        
        # Calculate histogram
        hist, bin_edges = np.histogram(arr, bins=20)
        for i in range(len(hist)):
            bin_key = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
            stats['histogram'][bin_key] = int(hist[i])
        
        return stats
    
    def _save_profile(self, source_id):
        """
        Save profile to file
        
        Args:
            source_id (str): Source identifier
        """
        if source_id not in self.profiles:
            return
        
        try:
            # Create profile file path
            profile_file = os.path.join(self.profile_dir, f"{source_id}_profile.json")
            
            # Save profile
            with open(profile_file, 'w') as f:
                json.dump(self.profiles[source_id], f, indent=2)
            
            self.logger.debug(f"Saved profile for {source_id} to {profile_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving profile for {source_id}: {str(e)}")
    
    def get_profile(self, source_id):
        """
        Get profile for a data source
        
        Args:
            source_id (str): Source identifier
            
        Returns:
            dict: Profile or None if source not found
        """
        if source_id not in self.profiles:
            self.logger.warning(f"Profile for {source_id} not found")
            return None
        
        return self.profiles[source_id].copy()
    
    def get_profile_history(self, source_id, count=None):
        """
        Get profile history for a data source
        
        Args:
            source_id (str): Source identifier
            count (int): Number of records to get, None means get all
            
        Returns:
            list: List of profile history records or None if source not found
        """
        if source_id not in self.profile_history:
            self.logger.warning(f"Profile history for {source_id} not found")
            return None
        
        if count is None:
            return self.profile_history[source_id].copy()
        else:
            return self.profile_history[source_id][-count:]
    
    def get_profile_metrics(self):
        """
        Get profile metrics
        
        Returns:
            dict: Profile metrics
        """
        return self.profile_metrics.copy()
    
    def compare_profiles(self, source_id1, source_id2):
        """
        Compare profiles of two data sources
        
        Args:
            source_id1 (str): First source identifier
            source_id2 (str): Second source identifier
            
        Returns:
            dict: Comparison result or None if sources not found
        """
        if source_id1 not in self.profiles:
            self.logger.warning(f"Profile for {source_id1} not found")
            return None
        
        if source_id2 not in self.profiles:
            self.logger.warning(f"Profile for {source_id2} not found")
            return None
        
        # Get profiles
        profile1 = self.profiles[source_id1]
        profile2 = self.profiles[source_id2]
        
        # Compare profiles
        comparison = {
            'source1': source_id1,
            'source2': source_id2,
            'common_fields': [],
            'unique_fields1': [],
            'unique_fields2': [],
            'field_comparisons': {}
        }
        
        # Get field sets
        fields1 = set(profile1['fields'].keys())
        fields2 = set(profile2['fields'].keys())
        
        # Find common and unique fields
        comparison['common_fields'] = list(fields1 & fields2)
        comparison['unique_fields1'] = list(fields1 - fields2)
        comparison['unique_fields2'] = list(fields2 - fields1)
        
        # Compare common fields
        for field in comparison['common_fields']:
            field1 = profile1['fields'][field]
            field2 = profile2['fields'][field]
            
            # Skip if either field has no statistics
            if field1['profile_count'] == 0 or field2['profile_count'] == 0:
                continue
            
            # Compare field statistics
            field_comparison = {
                'mean_diff': abs(field1['mean'] - field2['mean']),
                'std_diff': abs(field1['std'] - field2['std']),
                'min_diff': abs(field1['min'] - field2['min']),
                'max_diff': abs(field1['max'] - field2['max']),
                'median_diff': abs(field1['median'] - field2['median']),
                'mean_pct_diff': (
                    abs(field1['mean'] - field2['mean']) / ((field1['mean'] + field2['mean']) / 2) * 100
                    if (field1['mean'] + field2['mean']) != 0 else 0
                )
            }
            
            comparison['field_comparisons'][field] = field_comparison
        
        return comparison
    
    def generate_profile_report(self, source_id, output_dir=None):
        """
        Generate profile report with visualizations
        
        Args:
            source_id (str): Source identifier
            output_dir (str): Output directory, None means use profile directory
            
        Returns:
            bool: Whether successfully generated report
        """
        if source_id not in self.profiles:
            self.logger.warning(f"Profile for {source_id} not found")
            return False
        
        # Set output directory
        if output_dir is None:
            output_dir = self.profile_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Get profile
            profile = self.profiles[source_id]
            
            # Create report file
            report_file = os.path.join(output_dir, f"{source_id}_profile_report.txt")
            
            with open(report_file, 'w') as f:
                # Write header
                f.write(f"Data Source Profile Report\n")
                f.write(f"=========================\n\n")
                f.write(f"Source ID: {profile['source_id']}\n")
                f.write(f"Source Name: {profile['source_name']}\n")
                f.write(f"Created At: {profile['created_at']}\n")
                f.write(f"Updated At: {profile['updated_at']}\n\n")
                
                # Write field profiles
                f.write(f"Field Profiles\n")
                f.write(f"==============\n\n")
                
                for field, field_profile in profile['fields'].items():
                    f.write(f"Field: {field}\n")
                    f.write(f"---------\n")
                    f.write(f"Profile Count: {field_profile['profile_count']}\n")
                    f.write(f"Min: {field_profile['min']}\n")
                    f.write(f"Max: {field_profile['max']}\n")
                    f.write(f"Mean: {field_profile['mean']}\n")
                    f.write(f"Median: {field_profile['median']}\n")
                    f.write(f"Standard Deviation: {field_profile['std']}\n")
                    f.write(f"Variance: {field_profile['variance']}\n")
                    f.write(f"Skewness: {field_profile['skewness']}\n")
                    f.write(f"Kurtosis: {field_profile['kurtosis']}\n")
                    
                    f.write(f"\nPercentiles:\n")
                    for p, value in field_profile['percentiles'].items():
                        f.write(f"  {p}th: {value}\n")
                    
                    f.write(f"\nHistogram:\n")
                    for bin_key, count in field_profile['histogram'].items():
                        f.write(f"  {bin_key}: {count}\n")
                    
                    f.write("\n")
            
            # Generate visualizations
            self._generate_profile_visualizations(source_id, output_dir)
            
            self.logger.info(f"Generated profile report for {source_id} in {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating profile report for {source_id}: {str(e)}")
            return False
    
    def _generate_profile_visualizations(self, source_id, output_dir):
        """
        Generate profile visualizations
        
        Args:
            source_id (str): Source identifier
            output_dir (str): Output directory
        """
        if source_id not in self.profiles:
            return
        
        # Get profile
        profile = self.profiles[source_id]
        
        # Create figure with subplots
        num_fields = len(profile['fields'])
        if num_fields == 0:
            return
        
        # Determine grid size
        cols = min(3, num_fields)
        rows = (num_fields + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        
        # Flatten axes array if needed
        if rows > 1:
            axes = axes.flatten()
        elif num_fields == 1:
            axes = [axes]
        
        # Plot each field
        for i, (field, field_profile) in enumerate(profile['fields'].items()):
            # Skip if no histogram data
            if not field_profile['histogram']:
                continue
            
            # Parse histogram bins and counts
            bins = []
            counts = []
            
            for bin_key, count in field_profile['histogram'].items():
                bin_start, bin_end = map(float, bin_key.split('-'))
                bins.append(bin_start)
                counts.append(count)
            
            bins.append(bin_end)  # Add last bin edge
            
            # Plot histogram
            axes[i].hist(counts, bins=bins, alpha=0.7)
            axes[i].set_title(f"Field: {field}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
            
            # Add vertical lines for percentiles
            for p in [25, 50, 75]:
                if p in field_profile['percentiles']:
                    axes[i].axvline(
                        x=field_profile['percentiles'][p],
                        color='r',
                        linestyle='--',
                        label=f"{p}th percentile"
                    )
            
            # Add legend
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(num_fields, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_file = os.path.join(output_dir, f"{source_id}_profile_visualization.png")
        plt.savefig(fig_file)
        plt.close()
        
        self.logger.debug(f"Generated profile visualization for {source_id} in {fig_file}")
    
    def reset_profile(self, source_id):
        """
        Reset profile for a data source
        
        Args:
            source_id (str): Source identifier
            
        Returns:
            bool: Whether successfully reset profile
        """
        if source_id not in self.profiles:
            self.logger.warning(f"Profile for {source_id} not found")
            return False
        
        # Reset profile
        self.profiles[source_id] = {
            'source_id': source_id,
            'source_name': self.data_sources[source_id]['source_name'],
            'fields': {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Reset history
        self.profile_history[source_id] = []
        
        self.logger.info(f"Reset profile for {source_id}")
        return True
