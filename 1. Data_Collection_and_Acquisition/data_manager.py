"""
Data Manager for Data Collection and Acquisition Section

Manages processed and validated data from previous modules, handles data update requests,
and provides managed data storage with retrieval responses and update logs.
"""

import logging
import threading
import sqlite3
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import gzip
import os
from enum import Enum

class DataType(Enum):
    TICK = "tick"
    OHLCV = "ohlcv"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    SYMBOL_INFO = "symbol_info"
    MARKET_STATUS = "market_status"

class StorageFormat(Enum):
    JSON = "json"
    PICKLE = "pickle"
    PARQUET = "parquet"
    CSV = "csv"
    HDF5 = "hdf5"

@dataclass
class DataRecord:
    """Data record with metadata"""
    id: str
    symbol: str
    data_type: DataType
    timestamp: datetime
    data: Any
    source: str
    metadata: Dict[str, Any]
    checksum: str
    size_bytes: int
    compressed: bool = False

@dataclass
class DataQuery:
    """Data query parameters"""
    symbols: Optional[List[str]] = None
    data_types: Optional[List[DataType]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sources: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: int = 0
    order_by: str = "timestamp"
    ascending: bool = True

class DataCollectionManager:
    """
    Comprehensive data management for the Data Collection and Acquisition section
    """
    
    def __init__(self, storage_path: str = "./data/collection", 
                 db_path: str = "./data/collection.db",
                 max_memory_size: int = 1024 * 1024 * 1024):  # 1GB
        """
        Initialize Data Collection Manager
        
        Args:
            storage_path: Path for data storage
            db_path: Path for metadata database
            max_memory_size: Maximum memory usage in bytes
        """
        self.logger = logging.getLogger(__name__)
        self.storage_path = Path(storage_path)
        self.db_path = db_path
        self.max_memory_size = max_memory_size
        
        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.memory_cache: Dict[str, DataRecord] = {}
        self.memory_usage = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Threading
        self.lock = threading.RLock()
        
        # Callbacks
        self.update_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'records_stored': 0,
            'records_retrieved': 0,
            'total_size_bytes': 0,
            'compression_ratio': 0.0,
            'last_update': None
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_records (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT,
                    checksum TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    compressed BOOLEAN NOT NULL,
                    file_path TEXT,
                    storage_format TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON data_records(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON data_records(data_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON data_records(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON data_records(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON data_records(created_at)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def store_data(self, symbol: str, data_type: DataType, data: Any, 
                   source: str, metadata: Optional[Dict] = None,
                   storage_format: StorageFormat = StorageFormat.PICKLE,
                   compress: bool = True) -> str:
        """
        Store data with automatic management
        
        Args:
            symbol: Trading symbol
            data_type: Type of data
            data: Data to store
            source: Data source identifier
            metadata: Additional metadata
            storage_format: Storage format
            compress: Whether to compress data
            
        Returns:
            str: Record ID
        """
        try:
            with self.lock:
                # Generate record ID
                timestamp = datetime.now()
                record_id = self._generate_record_id(symbol, data_type, timestamp, source)
                
                # Prepare metadata
                if metadata is None:
                    metadata = {}
                metadata.update({
                    'original_type': str(type(data)),
                    'shape': getattr(data, 'shape', None),
                    'columns': list(data.columns) if hasattr(data, 'columns') else None
                })
                
                # Serialize data
                serialized_data = self._serialize_data(data, storage_format)
                
                # Compress if requested
                if compress:
                    serialized_data = gzip.compress(serialized_data)
                
                # Calculate checksum
                checksum = hashlib.sha256(serialized_data).hexdigest()
                
                # Create record
                record = DataRecord(
                    id=record_id,
                    symbol=symbol,
                    data_type=data_type,
                    timestamp=timestamp,
                    data=data,  # Keep original data in memory
                    source=source,
                    metadata=metadata,
                    checksum=checksum,
                    size_bytes=len(serialized_data),
                    compressed=compress
                )
                
                # Store to file
                file_path = self._get_file_path(record_id, storage_format)
                with open(file_path, 'wb') as f:
                    f.write(serialized_data)
                
                # Store metadata in database
                self._store_metadata(record, file_path, storage_format)
                
                # Add to memory cache if space allows
                if self.memory_usage + record.size_bytes <= self.max_memory_size:
                    self.memory_cache[record_id] = record
                    self.memory_usage += record.size_bytes
                else:
                    # Remove oldest records from cache
                    self._cleanup_memory_cache()
                    if self.memory_usage + record.size_bytes <= self.max_memory_size:
                        self.memory_cache[record_id] = record
                        self.memory_usage += record.size_bytes
                
                # Update statistics
                self.stats['records_stored'] += 1
                self.stats['total_size_bytes'] += record.size_bytes
                self.stats['last_update'] = timestamp
                
                # Trigger callbacks
                self._trigger_update_callbacks('store', record)
                
                self.logger.debug(f"Stored data record {record_id} for {symbol}")
                return record_id
                
        except Exception as e:
            self.logger.error(f"Failed to store data: {str(e)}")
            raise
    
    def retrieve_data(self, record_id: str) -> Optional[DataRecord]:
        """
        Retrieve data by record ID
        
        Args:
            record_id: Record identifier
            
        Returns:
            DataRecord: Retrieved record or None if not found
        """
        try:
            with self.lock:
                # Check memory cache first
                if record_id in self.memory_cache:
                    self.cache_hits += 1
                    self.stats['records_retrieved'] += 1
                    return self.memory_cache[record_id]
                
                self.cache_misses += 1
                
                # Query database for metadata
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM data_records WHERE id = ?
                ''', (record_id,))
                
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    return None
                
                # Load data from file
                file_path = row[9]  # file_path column
                storage_format = StorageFormat(row[10])  # storage_format column
                
                if not os.path.exists(file_path):
                    self.logger.error(f"Data file not found: {file_path}")
                    return None
                
                with open(file_path, 'rb') as f:
                    serialized_data = f.read()
                
                # Decompress if needed
                if row[8]:  # compressed column
                    serialized_data = gzip.decompress(serialized_data)
                
                # Deserialize data
                data = self._deserialize_data(serialized_data, storage_format)
                
                # Create record
                record = DataRecord(
                    id=row[0],
                    symbol=row[1],
                    data_type=DataType(row[2]),
                    timestamp=datetime.fromtimestamp(row[3]),
                    data=data,
                    source=row[4],
                    metadata=json.loads(row[5]) if row[5] else {},
                    checksum=row[6],
                    size_bytes=row[7],
                    compressed=row[8]
                )
                
                # Add to cache if space allows
                if self.memory_usage + record.size_bytes <= self.max_memory_size:
                    self.memory_cache[record_id] = record
                    self.memory_usage += record.size_bytes
                
                self.stats['records_retrieved'] += 1
                return record
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve data {record_id}: {str(e)}")
            return None
    
    def query_data(self, query: DataQuery) -> List[DataRecord]:
        """
        Query data based on criteria
        
        Args:
            query: Query parameters
            
        Returns:
            List[DataRecord]: Matching records
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Build query
                where_conditions = []
                params = []
                
                if query.symbols:
                    placeholders = ','.join('?' * len(query.symbols))
                    where_conditions.append(f'symbol IN ({placeholders})')
                    params.extend(query.symbols)
                
                if query.data_types:
                    placeholders = ','.join('?' * len(query.data_types))
                    where_conditions.append(f'data_type IN ({placeholders})')
                    params.extend([dt.value for dt in query.data_types])
                
                if query.start_time:
                    where_conditions.append('timestamp >= ?')
                    params.append(query.start_time.timestamp())
                
                if query.end_time:
                    where_conditions.append('timestamp <= ?')
                    params.append(query.end_time.timestamp())
                
                if query.sources:
                    placeholders = ','.join('?' * len(query.sources))
                    where_conditions.append(f'source IN ({placeholders})')
                    params.extend(query.sources)
                
                # Construct SQL
                sql = 'SELECT id FROM data_records'
                if where_conditions:
                    sql += ' WHERE ' + ' AND '.join(where_conditions)
                
                sql += f' ORDER BY {query.order_by}'
                if not query.ascending:
                    sql += ' DESC'
                
                if query.limit:
                    sql += f' LIMIT {query.limit} OFFSET {query.offset}'
                
                # Execute query
                cursor.execute(sql, params)
                record_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                # Retrieve records
                records = []
                for record_id in record_ids:
                    record = self.retrieve_data(record_id)
                    if record:
                        records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to query data: {str(e)}")
            return []
    
    def delete_data(self, record_id: str) -> bool:
        """
        Delete data record
        
        Args:
            record_id: Record identifier
            
        Returns:
            bool: Success status
        """
        try:
            with self.lock:
                # Get file path from database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT file_path FROM data_records WHERE id = ?', (record_id,))
                row = cursor.fetchone()
                
                if not row:
                    conn.close()
                    return False
                
                file_path = row[0]
                
                # Delete from database
                cursor.execute('DELETE FROM data_records WHERE id = ?', (record_id,))
                conn.commit()
                conn.close()
                
                # Delete file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Remove from cache
                if record_id in self.memory_cache:
                    record = self.memory_cache[record_id]
                    self.memory_usage -= record.size_bytes
                    del self.memory_cache[record_id]
                
                self.logger.debug(f"Deleted data record {record_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete data {record_id}: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data management statistics"""
        with self.lock:
            cache_stats = {
                'cache_size': len(self.memory_cache),
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
            
            return {**self.stats, **cache_stats}
    
    def add_update_callback(self, callback: Callable):
        """Add callback for data updates"""
        self.update_callbacks.append(callback)
    
    def cleanup_old_data(self, older_than: timedelta):
        """
        Clean up data older than specified time
        
        Args:
            older_than: Delete data older than this
        """
        try:
            cutoff_time = datetime.now() - older_than
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get old records
            cursor.execute('''
                SELECT id, file_path FROM data_records 
                WHERE timestamp < ?
            ''', (cutoff_time.timestamp(),))
            
            old_records = cursor.fetchall()
            
            # Delete records
            for record_id, file_path in old_records:
                cursor.execute('DELETE FROM data_records WHERE id = ?', (record_id,))
                
                # Delete file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Remove from cache
                if record_id in self.memory_cache:
                    record = self.memory_cache[record_id]
                    self.memory_usage -= record.size_bytes
                    del self.memory_cache[record_id]
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {len(old_records)} old records")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
    
    def _generate_record_id(self, symbol: str, data_type: DataType, 
                           timestamp: datetime, source: str) -> str:
        """Generate unique record ID"""
        data_str = f"{symbol}_{data_type.value}_{timestamp.isoformat()}_{source}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _serialize_data(self, data: Any, format: StorageFormat) -> bytes:
        """Serialize data based on format"""
        if format == StorageFormat.PICKLE:
            return pickle.dumps(data)
        elif format == StorageFormat.JSON:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                return data.to_json().encode()
            else:
                return json.dumps(data, default=str).encode()
        elif format == StorageFormat.PARQUET:
            if isinstance(data, pd.DataFrame):
                return data.to_parquet()
            else:
                raise ValueError("Parquet format only supports DataFrames")
        elif format == StorageFormat.CSV:
            if isinstance(data, pd.DataFrame):
                return data.to_csv().encode()
            else:
                raise ValueError("CSV format only supports DataFrames")
        else:
            raise ValueError(f"Unsupported storage format: {format}")
    
    def _deserialize_data(self, data: bytes, format: StorageFormat) -> Any:
        """Deserialize data based on format"""
        if format == StorageFormat.PICKLE:
            return pickle.loads(data)
        elif format == StorageFormat.JSON:
            return json.loads(data.decode())
        elif format == StorageFormat.PARQUET:
            return pd.read_parquet(data)
        elif format == StorageFormat.CSV:
            return pd.read_csv(data.decode())
        else:
            raise ValueError(f"Unsupported storage format: {format}")
    
    def _get_file_path(self, record_id: str, format: StorageFormat) -> str:
        """Get file path for record"""
        extension = {
            StorageFormat.PICKLE: '.pkl',
            StorageFormat.JSON: '.json',
            StorageFormat.PARQUET: '.parquet',
            StorageFormat.CSV: '.csv',
            StorageFormat.HDF5: '.h5'
        }.get(format, '.dat')
        
        return str(self.storage_path / f"{record_id}{extension}")
    
    def _store_metadata(self, record: DataRecord, file_path: str, 
                       storage_format: StorageFormat):
        """Store record metadata in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_records 
            (id, symbol, data_type, timestamp, source, metadata, checksum, 
             size_bytes, compressed, file_path, storage_format, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.id,
            record.symbol,
            record.data_type.value,
            record.timestamp.timestamp(),
            record.source,
            json.dumps(record.metadata),
            record.checksum,
            record.size_bytes,
            record.compressed,
            file_path,
            storage_format.value,
            datetime.now().timestamp(),
            datetime.now().timestamp()
        ))
        
        conn.commit()
        conn.close()
    
    def _cleanup_memory_cache(self):
        """Remove oldest records from memory cache"""
        if not self.memory_cache:
            return
        
        # Sort by timestamp and remove oldest 25%
        sorted_records = sorted(self.memory_cache.items(), 
                              key=lambda x: x[1].timestamp)
        
        remove_count = max(1, len(sorted_records) // 4)
        
        for i in range(remove_count):
            record_id, record = sorted_records[i]
            self.memory_usage -= record.size_bytes
            del self.memory_cache[record_id]
    
    def _trigger_update_callbacks(self, operation: str, record: DataRecord):
        """Trigger update callbacks"""
        for callback in self.update_callbacks:
            try:
                callback(operation, record)
            except Exception as e:
                self.logger.error(f"Update callback error: {str(e)}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            # Close any open connections
            pass
        except:
            pass