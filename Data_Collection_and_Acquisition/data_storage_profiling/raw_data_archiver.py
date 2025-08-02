# raw_data_archiver.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
import os
import json
import gzip
import pickle
import shutil
from pathlib import Path

class RawDataArchiver:
    """
    A class to persistently store raw collected data for historical analysis, audit, and replay.
    Implements efficient compression, partitioning, and retention policies.
    """
    
    def __init__(self, archive_dir="archive/raw_data", compression=True, retention_days=365):
        """
        Initialize Raw Data Archiver
        
        Args:
            archive_dir (str): Directory for archived data
            compression (bool): Whether to compress archived data
            retention_days (int): Number of days to retain data
        """
        self.archive_dir = archive_dir
        self.compression = compression
        self.retention_days = retention_days
        self.logger = self._setup_logger()
        
        # Create archive directory if it doesn't exist
        os.makedirs(archive_dir, exist_ok=True)
        
        # Archive status
        self.is_archiving = False
        self.archive_thread = None
        
        # Archive metrics
        self.archive_metrics = {
            'total_archived': 0,
            'total_retrieved': 0,
            'total_size': 0,
            'compression_ratio': 0,
            'last_archive_time': None,
            'last_retrieval_time': None,
            'oldest_data': None,
            'newest_data': None
        }
        
        # Data streams
        self.data_streams = {}
        self.archive_queues = {}
        
        # Archive index
        self.archive_index_file = os.path.join(archive_dir, "archive_index.json")
        self.archive_index = self._load_archive_index()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("RawDataArchiver")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("raw_data_archiver.log")
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
    
    def _load_archive_index(self):
        """Load archive index from file"""
        if os.path.exists(self.archive_index_file):
            try:
                with open(self.archive_index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading archive index: {str(e)}")
        
        # Return empty index if file doesn't exist or error occurred
        return {
            'streams': {},
            'files': {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
    
    def _save_archive_index(self):
        """Save archive index to file"""
        try:
            # Update timestamp
            self.archive_index['updated_at'] = datetime.now().isoformat()
            
            # Save to file
            with open(self.archive_index_file, 'w') as f:
                json.dump(self.archive_index, f, indent=2)
            
            self.logger.debug("Saved archive index")
        except Exception as e:
            self.logger.error(f"Error saving archive index: {str(e)}")
    
    def add_data_stream(self, stream_id, stream_name=None, data_format='json'):
        """
        Add data stream for archiving
        
        Args:
            stream_id (str): Stream identifier
            stream_name (str): Stream display name
            data_format (str): Data format ('json', 'csv', 'parquet', 'pickle')
            
        Returns:
            bool: Whether successfully added stream
        """
        if stream_id in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} already exists")
            return False
        
        # Validate data format
        valid_formats = ['json', 'csv', 'parquet', 'pickle']
        if data_format not in valid_formats:
            self.logger.error(f"Invalid data format: {data_format}")
            return False
        
        # Add stream
        self.data_streams[stream_id] = {
            'stream_name': stream_name or stream_id,
            'data_format': data_format,
            'created_at': datetime.now()
        }
        
        # Add queue
        self.archive_queues[stream_id] = []
        
        # Update index
        self.archive_index['streams'][stream_id] = {
            'stream_name': stream_name or stream_id,
            'data_format': data_format,
            'created_at': datetime.now().isoformat(),
            'files': []
        }
        
        # Save index
        self._save_archive_index()
        
        # If this is the first stream, start the archive thread
        if not self.is_archiving:
            self.is_archiving = True
            self.archive_thread = threading.Thread(target=self._archive_data)
            self.archive_thread.daemon = True
            self.archive_thread.start()
        
        self.logger.info(f"Added data stream {stream_id} for archiving")
        return True
    
    def remove_data_stream(self, stream_id):
        """
        Remove data stream from archiving
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            bool: Whether successfully removed stream
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Archive remaining data before removing
        self._archive_stream_data(stream_id)
        
        # Remove stream and queue
        del self.data_streams[stream_id]
        del self.archive_queues[stream_id]
        
        # Update index
        if stream_id in self.archive_index['streams']:
            del self.archive_index['streams'][stream_id]
            self._save_archive_index()
        
        # If no more streams, stop the archive thread
        if not self.data_streams and self.is_archiving:
            self.is_archiving = False
            if self.archive_thread and self.archive_thread.is_alive():
                self.archive_thread.join(timeout=5)
        
        self.logger.info(f"Removed data stream {stream_id} from archiving")
        return True
    
    def set_compression(self, enabled):
        """
        Set whether compression is enabled
        
        Args:
            enabled (bool): Whether compression is enabled
            
        Returns:
            bool: Whether successfully set option
        """
        if not isinstance(enabled, bool):
            self.logger.error("Enabled must be a boolean")
            return False
        
        self.compression = enabled
        self.logger.info(f"Set compression to {enabled}")
        return True
    
    def set_retention_days(self, days):
        """
        Set retention days
        
        Args:
            days (int): Number of days to retain data
            
        Returns:
            bool: Whether successfully set retention
        """
        if not isinstance(days, int) or days <= 0:
            self.logger.error("Retention days must be a positive integer")
            return False
        
        self.retention_days = days
        self.logger.info(f"Set retention days to {days}")
        return True
    
    def archive_data(self, stream_id, data):
        """
        Archive data
        
        Args:
            stream_id (str): Stream identifier
            data (dict): Data to archive
            
        Returns:
            bool: Whether successfully archived data
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Add timestamp to data
        if isinstance(data, dict):
            data_copy = data.copy()
            data_copy['_archive_timestamp'] = datetime.now()
        else:
            data_copy = {
                'value': data,
                '_archive_timestamp': datetime.now()
            }
        
        # Add to queue
        self.archive_queues[stream_id].append(data_copy)
        
        # Limit queue size
        max_queue_size = 10000
        if len(self.archive_queues[stream_id]) > max_queue_size:
            self.archive_queues[stream_id] = self.archive_queues[stream_id][-max_queue_size:]
        
        return True
    
    def retrieve_archived_data(self, stream_id, start_time=None, end_time=None, count=None):
        """
        Retrieve archived data
        
        Args:
            stream_id (str): Stream identifier
            start_time (datetime): Start time for retrieval
            end_time (datetime): End time for retrieval
            count (int): Number of records to retrieve, None means all
            
        Returns:
            list: List of data records or None if stream not found
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return None
        
        # Get stream info
        stream_info = self.data_streams[stream_id]
        data_format = stream_info['data_format']
        
        # Get files for stream from index
        if stream_id not in self.archive_index['streams']:
            return []
        
        file_list = self.archive_index['streams'][stream_id]['files']
        
        # Filter files by time range
        filtered_files = []
        for file_info in file_list:
            file_start_time = datetime.fromisoformat(file_info['start_time'])
            file_end_time = datetime.fromisoformat(file_info['end_time'])
            
            # Skip if file is outside time range
            if start_time is not None and file_end_time < start_time:
                continue
            
            if end_time is not None and file_start_time > end_time:
                continue
            
            filtered_files.append(file_info)
        
        # Sort files by start time
        filtered_files.sort(key=lambda x: x['start_time'])
        
        # Retrieve data from files
        all_data = []
        for file_info in filtered_files:
            file_path = file_info['file_path']
            
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.warning(f"Archive file not found: {file_path}")
                continue
            
            # Load data based on format
            try:
                if data_format == 'json':
                    data = self._load_json_file(file_path)
                elif data_format == 'csv':
                    data = self._load_csv_file(file_path)
                elif data_format == 'parquet':
                    data = self._load_parquet_file(file_path)
                elif data_format == 'pickle':
                    data = self._load_pickle_file(file_path)
                else:
                    self.logger.error(f"Unsupported data format: {data_format}")
                    continue
                
                # Filter by time range
                filtered_data = []
                for record in data:
                    timestamp = record.get('_archive_timestamp')
                    if timestamp is None:
                        continue
                    
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    
                    if start_time is not None and timestamp < start_time:
                        continue
                    
                    if end_time is not None and timestamp > end_time:
                        continue
                    
                    filtered_data.append(record)
                
                all_data.extend(filtered_data)
                
            except Exception as e:
                self.logger.error(f"Error loading archive file {file_path}: {str(e)}")
        
        # Limit count if specified
        if count is not None:
            all_data = all_data[-count:]
        
        # Update metrics
        self.archive_metrics['total_retrieved'] += len(all_data)
        self.archive_metrics['last_retrieval_time'] = datetime.now()
        
        return all_data
    
    def retrieve_archived_data_dataframe(self, stream_id, start_time=None, end_time=None, count=None):
        """
        Retrieve archived data as DataFrame
        
        Args:
            stream_id (str): Stream identifier
            start_time (datetime): Start time for retrieval
            end_time (datetime): End time for retrieval
            count (int): Number of records to retrieve, None means all
            
        Returns:
            pandas.DataFrame: DataFrame containing data or None if stream not found
        """
        data = self.retrieve_archived_data(stream_id, start_time, end_time, count)
        
        if data is None:
            return None
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert timestamps
        if '_archive_timestamp' in df.columns:
            df['_archive_timestamp'] = pd.to_datetime(df['_archive_timestamp'])
        
        return df
    
    def _archive_data(self):
        """
        Internal method to archive data, runs in separate thread
        """
        while self.is_archiving:
            try:
                # Process each stream
                for stream_id in self.archive_queues:
                    # Archive data if queue is not empty
                    if self.archive_queues[stream_id]:
                        self._archive_stream_data(stream_id)
                
                # Clean up old files
                self._cleanup_old_files()
                
                # Sleep for a while
                time.sleep(60)  # Archive every minute
                
            except Exception as e:
                self.logger.error(f"Error archiving data: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _archive_stream_data(self, stream_id):
        """
        Archive data for a stream
        
        Args:
            stream_id (str): Stream identifier
        """
        if stream_id not in self.archive_queues or not self.archive_queues[stream_id]:
            return
        
        # Get data from queue
        data = self.archive_queues[stream_id].copy()
        self.archive_queues[stream_id] = []
        
        if not data:
            return
        
        # Get stream info
        stream_info = self.data_streams[stream_id]
        data_format = stream_info['data_format']
        
        # Determine file path
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        
        file_dir = os.path.join(self.archive_dir, stream_id, date_str)
        os.makedirs(file_dir, exist_ok=True)
        
        file_name = f"{stream_id}_{date_str}_{time_str}.{data_format}"
        if self.compression:
            file_name += ".gz"
        
        file_path = os.path.join(file_dir, file_name)
        
        # Save data based on format
        try:
            if data_format == 'json':
                self._save_json_file(file_path, data)
            elif data_format == 'csv':
                self._save_csv_file(file_path, data)
            elif data_format == 'parquet':
                self._save_parquet_file(file_path, data)
            elif data_format == 'pickle':
                self._save_pickle_file(file_path, data)
            else:
                self.logger.error(f"Unsupported data format: {data_format}")
                return
            
            # Get time range
            timestamps = [record.get('_archive_timestamp') for record in data]
            timestamps = [ts for ts in timestamps if ts is not None]
            
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
            else:
                start_time = now
                end_time = now
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Update index
            file_info = {
                'file_path': file_path,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'record_count': len(data),
                'file_size': file_size,
                'compressed': self.compression,
                'created_at': now.isoformat()
            }
            
            self.archive_index['streams'][stream_id]['files'].append(file_info)
            
            if file_path not in self.archive_index['files']:
                self.archive_index['files'][file_path] = file_info
            
            # Save index
            self._save_archive_index()
            
            # Update metrics
            self.archive_metrics['total_archived'] += len(data)
            self.archive_metrics['total_size'] += file_size
            self.archive_metrics['last_archive_time'] = now
            
            # Update oldest and newest data
            if self.archive_metrics['oldest_data'] is None or start_time < self.archive_metrics['oldest_data']:
                self.archive_metrics['oldest_data'] = start_time
            
            if self.archive_metrics['newest_data'] is None or end_time > self.archive_metrics['newest_data']:
                self.archive_metrics['newest_data'] = end_time
            
            # Calculate compression ratio
            if self.compression:
                # Estimate uncompressed size
                estimated_size = len(json.dumps(data)) if data_format == 'json' else len(str(data))
                self.archive_metrics['compression_ratio'] = (
                    (estimated_size - file_size) / estimated_size * 100
                )
            
            self.logger.info(f"Archived {len(data)} records for {stream_id} to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error archiving data for {stream_id}: {str(e)}")
    
    def _cleanup_old_files(self):
        """Clean up old files based on retention policy"""
        if self.retention_days <= 0:
            return
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Files to remove
        files_to_remove = []
        
        # Check each file in index
        for file_path, file_info in self.archive_index['files'].items():
            end_time = datetime.fromisoformat(file_info['end_time'])
            
            if end_time < cutoff_date:
                files_to_remove.append(file_path)
        
        # Remove files
        for file_path in files_to_remove:
            try:
                # Remove file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Remove from index
                if file_path in self.archive_index['files']:
                    del self.archive_index['files'][file_path]
                
                # Remove from stream files
                for stream_id, stream_info in self.archive_index['streams'].items():
                    stream_files = stream_info['files']
                    stream_files = [f for f in stream_files if f['file_path'] != file_path]
                    stream_info['files'] = stream_files
                
                self.logger.info(f"Removed old archive file: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error removing old archive file {file_path}: {str(e)}")
        
        # Save index if files were removed
        if files_to_remove:
            self._save_archive_index()
    
    def _save_json_file(self, file_path, data):
        """Save data as JSON file"""
        if self.compression:
            with gzip.open(file_path, 'wt') as f:
                json.dump(data, f)
        else:
            with open(file_path, 'w') as f:
                json.dump(data, f)
    
    def _load_json_file(self, file_path):
        """Load data from JSON file"""
        if self.compression:
            with gzip.open(file_path, 'rt') as f:
                return json.load(f)
        else:
            with open(file_path, 'r') as f:
                return json.load(f)
    
    def _save_csv_file(self, file_path, data):
        """Save data as CSV file"""
        df = pd.DataFrame(data)
        
        if self.compression:
            df.to_csv(file_path, index=False, compression='gzip')
        else:
            df.to_csv(file_path, index=False)
    
    def _load_csv_file(self, file_path):
        """Load data from CSV file"""
        if self.compression:
            return pd.read_csv(file_path, compression='gzip').to_dict('records')
        else:
            return pd.read_csv(file_path).to_dict('records')
    
    def _save_parquet_file(self, file_path, data):
        """Save data as Parquet file"""
        df = pd.DataFrame(data)
        
        if self.compression:
            df.to_parquet(file_path, index=False, compression='gzip')
        else:
            df.to_parquet(file_path, index=False)
    
    def _load_parquet_file(self, file_path):
        """Load data from Parquet file"""
        if self.compression:
            return pd.read_parquet(file_path).to_dict('records')
        else:
            return pd.read_parquet(file_path).to_dict('records')
    
    def _save_pickle_file(self, file_path, data):
        """Save data as Pickle file"""
        if self.compression:
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
    
    def _load_pickle_file(self, file_path):
        """Load data from Pickle file"""
        if self.compression:
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def get_archive_metrics(self):
        """
        Get archive metrics
        
        Returns:
            dict: Archive metrics
        """
        return self.archive_metrics.copy()
    
    def get_stream_info(self, stream_id):
        """
        Get stream information
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            dict: Stream information or None if stream not found
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return None
        
        # Get stream info
        stream_info = self.data_streams[stream_id].copy()
        
        # Add file count
        if stream_id in self.archive_index['streams']:
            stream_info['file_count'] = len(self.archive_index['streams'][stream_id]['files'])
        else:
            stream_info['file_count'] = 0
        
        # Add queue size
        stream_info['queue_size'] = len(self.archive_queues.get(stream_id, []))
        
        return stream_info
    
    def get_archive_file_list(self, stream_id=None):
        """
        Get list of archive files
        
        Args:
            stream_id (str): Stream identifier, None means all streams
            
        Returns:
            list: List of archive files
        """
        if stream_id is not None:
            if stream_id not in self.archive_index['streams']:
                return []
            
            return self.archive_index['streams'][stream_id]['files'].copy()
        else:
            # Return all files
            files = []
            for file_info in self.archive_index['files'].values():
                files.append(file_info.copy())
            
            return files
    
    def rebuild_archive_index(self):
        """
        Rebuild archive index by scanning archive directory
        
        Returns:
            bool: Whether successfully rebuilt index
        """
        try:
            # Reset index
            self.archive_index = {
                'streams': {},
                'files': {},
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Scan archive directory
            for root, dirs, files in os.walk(self.archive_dir):
                for file in files:
                    # Skip index file
                    if file == "archive_index.json":
                        continue
                    
                    # Get file path
                    file_path = os.path.join(root, file)
                    
                    # Get file info
                    file_stat = os.stat(file_path)
                    file_size = file_stat.st_size
                    
                    # Parse file name to get stream ID and timestamp
                    file_name = os.path.splitext(file)[0]
                    if file.endswith('.gz'):
                        file_name = os.path.splitext(file_name)[0]
                    
                    parts = file_name.split('_')
                    if len(parts) >= 3:
                        stream_id = parts[0]
                        date_str = parts[1]
                        time_str = parts[2]
                        
                        # Parse timestamp
                        try:
                            timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H%M%S")
                        except ValueError:
                            timestamp = datetime.fromtimestamp(file_stat.st_mtime)
                        
                        # Get data format
                        ext = os.path.splitext(file)[1].lower()[1:]  # Remove dot
                        if file.endswith('.gz'):
                            ext = os.path.splitext(os.path.splitext(file)[0])[1].lower()[1:]  # Remove dot
                        
                        # Add to index
                        file_info = {
                            'file_path': file_path,
                            'start_time': timestamp.isoformat(),
                            'end_time': timestamp.isoformat(),
                            'record_count': 0,  # Unknown without loading file
                            'file_size': file_size,
                            'compressed': file.endswith('.gz'),
                            'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                        }
                        
                        # Add to files
                        self.archive_index['files'][file_path] = file_info
                        
                        # Add to streams
                        if stream_id not in self.archive_index['streams']:
                            self.archive_index['streams'][stream_id] = {
                                'stream_name': stream_id,
                                'data_format': ext,
                                'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                                'files': []
                            }
                        
                        self.archive_index['streams'][stream_id]['files'].append(file_info)
            
            # Save index
            self._save_archive_index()
            
            self.logger.info("Rebuilt archive index")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebuilding archive index: {str(e)}")
            return False
    
    def export_archive(self, stream_id, export_dir, start_time=None, end_time=None, export_format='parquet'):
        """
        Export archived data to external directory
        
        Args:
            stream_id (str): Stream identifier
            export_dir (str): Export directory
            start_time (datetime): Start time for export
            end_time (datetime): End time for export
            export_format (str): Export format ('json', 'csv', 'parquet', 'pickle')
            
        Returns:
            bool: Whether successfully exported
        """
        if stream_id not in self.data_streams:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Validate export format
        valid_formats = ['json', 'csv', 'parquet', 'pickle']
        if export_format not in valid_formats:
            self.logger.error(f"Invalid export format: {export_format}")
            return False
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Get data
        data = self.retrieve_archived_data(stream_id, start_time, end_time)
        
        if not data:
            self.logger.warning(f"No data to export for stream {stream_id}")
            return False
        
        # Determine export file path
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        
        export_file = f"{stream_id}_export_{date_str}_{time_str}.{export_format}"
        export_path = os.path.join(export_dir, export_file)
        
        # Export data based on format
        try:
            if export_format == 'json':
                self._save_json_file(export_path, data)
            elif export_format == 'csv':
                self._save_csv_file(export_path, data)
            elif export_format == 'parquet':
                self._save_parquet_file(export_path, data)
            elif export_format == 'pickle':
                self._save_pickle_file(export_path, data)
            
            self.logger.info(f"Exported {len(data)} records for {stream_id} to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data for {stream_id}: {str(e)}")
            return False
