# data_buffer_fallback_storage.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
import logging
import os
import json
import pickle
import sqlite3
from pathlib import Path

class DataBufferFallbackStorage:
    """
    A class to temporarily store real-time data streams in memory before writing to disk.
    Provides fallback storage during system interruptions or network issues.
    """
    
    def __init__(self, buffer_size=10000, storage_dir="storage/buffer"):
        """
        Initialize Data Buffer Fallback Storage
        
        Args:
            buffer_size (int): Memory buffer size
            storage_dir (str): Directory for fallback storage
        """
        self.buffer_size = buffer_size
        self.storage_dir = storage_dir
        self.logger = self._setup_logger()
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Data buffers
        self.data_buffers = {}
        self.buffer_queues = {}
        
        # Storage status
        self.is_storing = False
        self.storage_thread = None
        
        # Fallback storage
        self.fallback_enabled = True
        self.fallback_file = os.path.join(storage_dir, "fallback_data.pkl")
        self.fallback_db = os.path.join(storage_dir, "fallback_data.db")
        
        # Storage metrics
        self.storage_metrics = {
            'total_stored': 0,
            'total_retrieved': 0,
            'buffer_hits': 0,
            'fallback_hits': 0,
            'last_storage_time': None,
            'last_retrieval_time': None
        }
        
        # Initialize fallback database
        self._init_fallback_db()
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("DataBufferFallbackStorage")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("data_buffer_fallback_storage.log")
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
    
    def _init_fallback_db(self):
        """Initialize fallback database"""
        try:
            conn = sqlite3.connect(self.fallback_db)
            cursor = conn.cursor()
            
            # Create table for data storage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fallback_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    data BLOB NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster retrieval
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stream_timestamp 
                ON fallback_data(stream_id, timestamp)
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Initialized fallback database: {self.fallback_db}")
        except Exception as e:
            self.logger.error(f"Error initializing fallback database: {str(e)}")
    
    def add_data_stream(self, stream_id, stream_name=None):
        """
        Add data stream for buffering
        
        Args:
            stream_id (str): Stream identifier
            stream_name (str): Stream display name
            
        Returns:
            bool: Whether successfully added stream
        """
        if stream_id in self.data_buffers:
            self.logger.warning(f"Data stream {stream_id} already exists")
            return False
        
        # Add buffer and queue
        self.data_buffers[stream_id] = []
        self.buffer_queues[stream_id] = queue.Queue(maxsize=self.buffer_size)
        
        # If this is the first stream, start the storage thread
        if not self.is_storing:
            self.is_storing = True
            self.storage_thread = threading.Thread(target=self._store_data)
            self.storage_thread.daemon = True
            self.storage_thread.start()
        
        self.logger.info(f"Added data stream {stream_id} for buffering")
        return True
    
    def remove_data_stream(self, stream_id):
        """
        Remove data stream from buffering
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            bool: Whether successfully removed stream
        """
        if stream_id not in self.data_buffers:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Save remaining data before removing
        self._save_stream_data(stream_id)
        
        # Remove buffer and queue
        del self.data_buffers[stream_id]
        del self.buffer_queues[stream_id]
        
        # If no more streams, stop the storage thread
        if not self.data_buffers and self.is_storing:
            self.is_storing = False
            if self.storage_thread and self.storage_thread.is_alive():
                self.storage_thread.join(timeout=5)
        
        self.logger.info(f"Removed data stream {stream_id} from buffering")
        return True
    
    def set_fallback_enabled(self, enabled):
        """
        Set whether fallback storage is enabled
        
        Args:
            enabled (bool): Whether fallback storage is enabled
            
        Returns:
            bool: Whether successfully set option
        """
        if not isinstance(enabled, bool):
            self.logger.error("Enabled must be a boolean")
            return False
        
        self.fallback_enabled = enabled
        self.logger.info(f"Set fallback storage to {enabled}")
        return True
    
    def store_data(self, stream_id, data):
        """
        Store data in buffer
        
        Args:
            stream_id (str): Stream identifier
            data (dict): Data to store
            
        Returns:
            bool: Whether successfully stored data
        """
        if stream_id not in self.data_buffers:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Add timestamp to data
        if isinstance(data, dict):
            data_copy = data.copy()
            data_copy['_buffer_timestamp'] = datetime.now()
        else:
            data_copy = {
                'value': data,
                '_buffer_timestamp': datetime.now()
            }
        
        # Add to buffer
        self.data_buffers[stream_id].append(data_copy)
        
        # Add to queue for storage
        try:
            self.buffer_queues[stream_id].put_nowait(data_copy)
        except queue.Full:
            # Buffer is full, remove oldest data
            try:
                self.buffer_queues[stream_id].get_nowait()
                self.buffer_queues[stream_id].put_nowait(data_copy)
            except queue.Empty:
                pass
        
        # Limit buffer size
        if len(self.data_buffers[stream_id]) > self.buffer_size:
            self.data_buffers[stream_id] = self.data_buffers[stream_id][-self.buffer_size:]
        
        # Update metrics
        self.storage_metrics['total_stored'] += 1
        self.storage_metrics['last_storage_time'] = datetime.now()
        
        return True
    
    def retrieve_data(self, stream_id, count=None, start_time=None, end_time=None):
        """
        Retrieve data from buffer
        
        Args:
            stream_id (str): Stream identifier
            count (int): Number of records to retrieve, None means all
            start_time (datetime): Start time for retrieval
            end_time (datetime): End time for retrieval
            
        Returns:
            list: List of data records or None if stream not found
        """
        if stream_id not in self.data_buffers:
            self.logger.warning(f"Data stream {stream_id} not found")
            return None
        
        # Get data from buffer
        data = self.data_buffers[stream_id].copy()
        
        # Filter by time range if specified
        if start_time is not None or end_time is not None:
            filtered_data = []
            
            for record in data:
                timestamp = record.get('_buffer_timestamp')
                if timestamp is None:
                    continue
                
                if start_time is not None and timestamp < start_time:
                    continue
                
                if end_time is not None and timestamp > end_time:
                    continue
                
                filtered_data.append(record)
            
            data = filtered_data
        
        # Limit count if specified
        if count is not None:
            data = data[-count:]
        
        # If no data in buffer, try fallback storage
        if not data and self.fallback_enabled:
            fallback_data = self._retrieve_from_fallback(stream_id, count, start_time, end_time)
            if fallback_data:
                data = fallback_data
                self.storage_metrics['fallback_hits'] += 1
                self.logger.info(f"Retrieved {len(data)} records from fallback storage for {stream_id}")
            else:
                self.logger.warning(f"No data found in buffer or fallback storage for {stream_id}")
        else:
            self.storage_metrics['buffer_hits'] += 1
        
        # Update metrics
        self.storage_metrics['total_retrieved'] += len(data)
        self.storage_metrics['last_retrieval_time'] = datetime.now()
        
        return data
    
    def retrieve_data_dataframe(self, stream_id, count=None, start_time=None, end_time=None):
        """
        Retrieve data from buffer as DataFrame
        
        Args:
            stream_id (str): Stream identifier
            count (int): Number of records to retrieve, None means all
            start_time (datetime): Start time for retrieval
            end_time (datetime): End time for retrieval
            
        Returns:
            pandas.DataFrame: DataFrame containing data or None if stream not found
        """
        data = self.retrieve_data(stream_id, count, start_time, end_time)
        
        if data is None:
            return None
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert timestamps
        if '_buffer_timestamp' in df.columns:
            df['_buffer_timestamp'] = pd.to_datetime(df['_buffer_timestamp'])
        
        return df
    
    def _store_data(self):
        """
        Internal method to store data, runs in separate thread
        """
        while self.is_storing:
            try:
                # Process each stream
                for stream_id in self.buffer_queues:
                    # Process all data in queue
                    while not self.buffer_queues[stream_id].empty():
                        try:
                            # Get data from queue
                            data = self.buffer_queues[stream_id].get_nowait()
                            
                            # Store in fallback if enabled
                            if self.fallback_enabled:
                                self._store_to_fallback(stream_id, data)
                            
                        except queue.Empty:
                            break
                
                # Sleep for a while
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error storing data: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _store_to_fallback(self, stream_id, data):
        """
        Store data to fallback storage
        
        Args:
            stream_id (str): Stream identifier
            data (dict): Data to store
        """
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            
            # Store in database
            conn = sqlite3.connect(self.fallback_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fallback_data (stream_id, timestamp, data)
                VALUES (?, ?, ?)
            ''', (stream_id, datetime.now(), serialized_data))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing data to fallback: {str(e)}")
    
    def _retrieve_from_fallback(self, stream_id, count=None, start_time=None, end_time=None):
        """
        Retrieve data from fallback storage
        
        Args:
            stream_id (str): Stream identifier
            count (int): Number of records to retrieve, None means all
            start_time (datetime): Start time for retrieval
            end_time (datetime): End time for retrieval
            
        Returns:
            list: List of data records
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.fallback_db)
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT data FROM fallback_data WHERE stream_id = ?"
            params = [stream_id]
            
            # Add time range filters if specified
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            # Add order by
            query += " ORDER BY timestamp DESC"
            
            # Add limit if specified
            if count is not None:
                query += " LIMIT ?"
                params.append(count)
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Close connection
            conn.close()
            
            # Deserialize data
            data = []
            for row in rows:
                try:
                    deserialized_data = pickle.loads(row[0])
                    data.append(deserialized_data)
                except Exception as e:
                    self.logger.error(f"Error deserializing data: {str(e)}")
            
            # Reverse to get chronological order
            data.reverse()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving data from fallback: {str(e)}")
            return []
    
    def _save_stream_data(self, stream_id):
        """
        Save remaining data for a stream
        
        Args:
            stream_id (str): Stream identifier
        """
        if stream_id not in self.data_buffers:
            return
        
        # Get remaining data
        data = self.data_buffers[stream_id]
        
        if not data:
            return
        
        # Save to fallback if enabled
        if self.fallback_enabled:
            for record in data:
                self._store_to_fallback(stream_id, record)
        
        # Clear buffer
        self.data_buffers[stream_id] = []
        
        self.logger.info(f"Saved remaining data for stream {stream_id}")
    
    def get_storage_metrics(self):
        """
        Get storage metrics
        
        Returns:
            dict: Storage metrics
        """
        return self.storage_metrics.copy()
    
    def get_buffer_info(self, stream_id):
        """
        Get buffer information for a stream
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            dict: Buffer information or None if stream not found
        """
        if stream_id not in self.data_buffers:
            self.logger.warning(f"Data stream {stream_id} not found")
            return None
        
        return {
            'stream_id': stream_id,
            'buffer_size': len(self.data_buffers[stream_id]),
            'max_buffer_size': self.buffer_size,
            'queue_size': self.buffer_queues[stream_id].qsize(),
            'utilization': len(self.data_buffers[stream_id]) / self.buffer_size
        }
    
    def save_buffer_to_csv(self, stream_id, filename, count=None, start_time=None, end_time=None):
        """
        Save buffer data to CSV file
        
        Args:
            stream_id (str): Stream identifier
            filename (str): Filename
            count (int): Number of records to save, None means all
            start_time (datetime): Start time for retrieval
            end_time (datetime): End time for retrieval
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.retrieve_data_dataframe(stream_id, count, start_time, end_time)
            
            if df is None or df.empty:
                self.logger.warning(f"No buffer data to save for stream {stream_id}")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} buffer records for {stream_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save buffer data for {stream_id} to CSV: {str(e)}")
            return False
    
    def save_buffer_to_parquet(self, stream_id, filename, count=None, start_time=None, end_time=None):
        """
        Save buffer data to Parquet file
        
        Args:
            stream_id (str): Stream identifier
            filename (str): Filename
            count (int): Number of records to save, None means all
            start_time (datetime): Start time for retrieval
            end_time (datetime): End time for retrieval
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.retrieve_data_dataframe(stream_id, count, start_time, end_time)
            
            if df is None or df.empty:
                self.logger.warning(f"No buffer data to save for stream {stream_id}")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} buffer records for {stream_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save buffer data for {stream_id} to Parquet: {str(e)}")
            return False
    
    def clear_buffer(self, stream_id):
        """
        Clear buffer for a stream
        
        Args:
            stream_id (str): Stream identifier
            
        Returns:
            bool: Whether successfully cleared buffer
        """
        if stream_id not in self.data_buffers:
            self.logger.warning(f"Data stream {stream_id} not found")
            return False
        
        # Save data before clearing
        self._save_stream_data(stream_id)
        
        # Clear buffer
        self.data_buffers[stream_id] = []
        
        # Clear queue
        while not self.buffer_queues[stream_id].empty():
            try:
                self.buffer_queues[stream_id].get_nowait()
            except queue.Empty:
                pass
        
        self.logger.info(f"Cleared buffer for stream {stream_id}")
        return True
    
    def clear_fallback_storage(self, stream_id=None):
        """
        Clear fallback storage
        
        Args:
            stream_id (str): Stream identifier, None means clear all
            
        Returns:
            int: Number of records cleared
        """
        try:
            conn = sqlite3.connect(self.fallback_db)
            cursor = conn.cursor()
            
            if stream_id is not None:
                # Clear specific stream
                cursor.execute("DELETE FROM fallback_data WHERE stream_id = ?", (stream_id,))
                count = cursor.rowcount
                self.logger.info(f"Cleared {count} fallback records for stream {stream_id}")
            else:
                # Clear all streams
                cursor.execute("DELETE FROM fallback_data")
                count = cursor.rowcount
                self.logger.info(f"Cleared {count} fallback records for all streams")
            
            conn.commit()
            conn.close()
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error clearing fallback storage: {str(e)}")
            return 0
