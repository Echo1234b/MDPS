#!/usr/bin/env python3
"""
MDPS Database Manager
Centralized data storage with optimization, versioning, and backup management
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import sqlite3
import aiosqlite
import pickle
import gzip
import hashlib

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Central database manager for MDPS system"""
    
    def __init__(self, host: str = "localhost", port: int = 5432, 
                 database: str = "mdps_db", username: str = "mdps_user", 
                 password: str = ""):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        
        # SQLite fallback for local development
        self.use_sqlite = host == "localhost" and port == 5432
        self.db_path = Path("data/mdps.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Connection pool
        self.connections = []
        self.max_connections = 10
        self.connection_lock = asyncio.Lock()
        
        # Cache
        self.cache = {}
        self.cache_size = 0
        self.max_cache_size = 100 * 1024 * 1024  # 100MB
        
    async def connect(self):
        """Connect to database"""
        try:
            if self.use_sqlite:
                await self._connect_sqlite()
            else:
                await self._connect_postgres()
                
            # Initialize database schema
            await self._initialize_schema()
            
            logger.info("Database connected successfully")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    async def _connect_sqlite(self):
        """Connect to SQLite database"""
        try:
            # Create connection pool
            for _ in range(self.max_connections):
                conn = await aiosqlite.connect(self.db_path)
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")
                await conn.execute("PRAGMA cache_size=10000")
                await conn.execute("PRAGMA temp_store=MEMORY")
                self.connections.append(conn)
                
            logger.info(f"SQLite connection pool created with {self.max_connections} connections")
            
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            raise
            
    async def _connect_postgres(self):
        """Connect to PostgreSQL database"""
        try:
            # This would be implemented with asyncpg or similar
            # For now, we'll use SQLite as fallback
            logger.warning("PostgreSQL not implemented, using SQLite fallback")
            await self._connect_sqlite()
            
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
            
    async def _initialize_schema(self):
        """Initialize database schema"""
        try:
            conn = await self._get_connection()
            
            # Create tables
            await self._create_tables(conn)
            
            # Create indexes
            await self._create_indexes(conn)
            
            await self._release_connection(conn)
            
            logger.info("Database schema initialized")
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise
            
    async def _create_tables(self, conn):
        """Create database tables"""
        tables = [
            # Market data tables
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timeframe TEXT,
                source TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(symbol, timestamp, timeframe)
            )
            """,
            
            # Technical indicators
            """
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                indicator_name TEXT NOT NULL,
                value REAL,
                parameters TEXT,
                timeframe TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(symbol, timestamp, indicator_name, timeframe)
            )
            """,
            
            # ML predictions
            """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                model_name TEXT NOT NULL,
                prediction REAL,
                confidence REAL,
                features TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(symbol, timestamp, model_name)
            )
            """,
            
            # Trading signals
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                signal_type TEXT NOT NULL,
                direction TEXT,
                strength REAL,
                confidence REAL,
                source TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(symbol, timestamp, signal_type)
            )
            """,
            
            # System metrics
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL,
                unit TEXT,
                tags TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
            """,
            
            # Alerts
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                source TEXT,
                acknowledged BOOLEAN DEFAULT FALSE,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            """,
            
            # Configuration
            """
            CREATE TABLE IF NOT EXISTS configuration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT,
                section TEXT,
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            """
        ]
        
        for table_sql in tables:
            await conn.execute(table_sql)
            
        await conn.commit()
        
    async def _create_indexes(self, conn):
        """Create database indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data(timeframe)",
            "CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol ON technical_indicators(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
            
        await conn.commit()
        
    async def _get_connection(self):
        """Get database connection from pool"""
        async with self.connection_lock:
            if self.connections:
                return self.connections.pop()
            else:
                # Create new connection if pool is empty
                conn = await aiosqlite.connect(self.db_path)
                await conn.execute("PRAGMA journal_mode=WAL")
                return conn
                
    async def _release_connection(self, conn):
        """Release connection back to pool"""
        async with self.connection_lock:
            if len(self.connections) < self.max_connections:
                self.connections.append(conn)
            else:
                await conn.close()
                
    async def disconnect(self):
        """Disconnect from database"""
        try:
            # Close all connections
            for conn in self.connections:
                await conn.close()
            self.connections.clear()
            
            logger.info("Database disconnected")
            
        except Exception as e:
            logger.error(f"Database disconnection failed: {e}")
            
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            conn = await self._get_connection()
            
            # Simple query to check connectivity
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            
            await self._release_connection(conn)
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
            
    async def store_market_data(self, symbol: str, timestamp: float, 
                               open_price: float, high: float, low: float, 
                               close: float, volume: float, timeframe: str, 
                               source: str = "default"):
        """Store market data"""
        try:
            conn = await self._get_connection()
            
            await conn.execute("""
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume, timeframe, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timestamp, open_price, high, low, close, volume, timeframe, source))
            
            await conn.commit()
            await self._release_connection(conn)
            
            # Update cache
            self._update_cache(f"market_data_{symbol}_{timeframe}", {
                'symbol': symbol,
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'timeframe': timeframe,
                'source': source
            })
            
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            raise
            
    async def get_market_data(self, symbol: str, timeframe: str, 
                             start_time: float = None, end_time: float = None, 
                             limit: int = 1000) -> List[Dict]:
        """Get market data"""
        try:
            # Check cache first
            cache_key = f"market_data_{symbol}_{timeframe}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if start_time is None or cached_data['timestamp'] >= start_time:
                    return [cached_data]
                    
            conn = await self._get_connection()
            
            query = """
                SELECT symbol, timestamp, open, high, low, close, volume, timeframe, source
                FROM market_data 
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
                
            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            
            await self._release_connection(conn)
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                data.append({
                    'symbol': row[0],
                    'timestamp': row[1],
                    'open': row[2],
                    'high': row[3],
                    'low': row[4],
                    'close': row[5],
                    'volume': row[6],
                    'timeframe': row[7],
                    'source': row[8]
                })
                
            return data
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return []
            
    async def store_technical_indicator(self, symbol: str, timestamp: float, 
                                      indicator_name: str, value: float, 
                                      parameters: Dict = None, timeframe: str = "1m"):
        """Store technical indicator"""
        try:
            conn = await self._get_connection()
            
            params_json = json.dumps(parameters) if parameters else None
            
            await conn.execute("""
                INSERT OR REPLACE INTO technical_indicators 
                (symbol, timestamp, indicator_name, value, parameters, timeframe)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, timestamp, indicator_name, value, params_json, timeframe))
            
            await conn.commit()
            await self._release_connection(conn)
            
        except Exception as e:
            logger.error(f"Failed to store technical indicator: {e}")
            raise
            
    async def get_technical_indicator(self, symbol: str, indicator_name: str, 
                                    timeframe: str = "1m", limit: int = 1000) -> List[Dict]:
        """Get technical indicator data"""
        try:
            conn = await self._get_connection()
            
            cursor = await conn.execute("""
                SELECT symbol, timestamp, indicator_name, value, parameters, timeframe
                FROM technical_indicators 
                WHERE symbol = ? AND indicator_name = ? AND timeframe = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (symbol, indicator_name, timeframe, limit))
            
            rows = await cursor.fetchall()
            await self._release_connection(conn)
            
            data = []
            for row in rows:
                data.append({
                    'symbol': row[0],
                    'timestamp': row[1],
                    'indicator_name': row[2],
                    'value': row[3],
                    'parameters': json.loads(row[4]) if row[4] else None,
                    'timeframe': row[5]
                })
                
            return data
            
        except Exception as e:
            logger.error(f"Failed to get technical indicator: {e}")
            return []
            
    async def store_ml_prediction(self, symbol: str, timestamp: float, 
                                model_name: str, prediction: float, 
                                confidence: float, features: Dict = None):
        """Store ML prediction"""
        try:
            conn = await self._get_connection()
            
            features_json = json.dumps(features) if features else None
            
            await conn.execute("""
                INSERT OR REPLACE INTO ml_predictions 
                (symbol, timestamp, model_name, prediction, confidence, features)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, timestamp, model_name, prediction, confidence, features_json))
            
            await conn.commit()
            await self._release_connection(conn)
            
        except Exception as e:
            logger.error(f"Failed to store ML prediction: {e}")
            raise
            
    async def get_ml_predictions(self, symbol: str, model_name: str = None, 
                                limit: int = 1000) -> List[Dict]:
        """Get ML predictions"""
        try:
            conn = await self._get_connection()
            
            if model_name:
                cursor = await conn.execute("""
                    SELECT symbol, timestamp, model_name, prediction, confidence, features
                    FROM ml_predictions 
                    WHERE symbol = ? AND model_name = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, model_name, limit))
            else:
                cursor = await conn.execute("""
                    SELECT symbol, timestamp, model_name, prediction, confidence, features
                    FROM ml_predictions 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, limit))
                
            rows = await cursor.fetchall()
            await self._release_connection(conn)
            
            data = []
            for row in rows:
                data.append({
                    'symbol': row[0],
                    'timestamp': row[1],
                    'model_name': row[2],
                    'prediction': row[3],
                    'confidence': row[4],
                    'features': json.loads(row[5]) if row[5] else None
                })
                
            return data
            
        except Exception as e:
            logger.error(f"Failed to get ML predictions: {e}")
            return []
            
    async def store_trading_signal(self, symbol: str, timestamp: float, 
                                 signal_type: str, direction: str = None, 
                                 strength: float = None, confidence: float = None, 
                                 source: str = "default"):
        """Store trading signal"""
        try:
            conn = await self._get_connection()
            
            await conn.execute("""
                INSERT OR REPLACE INTO trading_signals 
                (symbol, timestamp, signal_type, direction, strength, confidence, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timestamp, signal_type, direction, strength, confidence, source))
            
            await conn.commit()
            await self._release_connection(conn)
            
        except Exception as e:
            logger.error(f"Failed to store trading signal: {e}")
            raise
            
    async def get_trading_signals(self, symbol: str, signal_type: str = None, 
                                 limit: int = 1000) -> List[Dict]:
        """Get trading signals"""
        try:
            conn = await self._get_connection()
            
            if signal_type:
                cursor = await conn.execute("""
                    SELECT symbol, timestamp, signal_type, direction, strength, confidence, source
                    FROM trading_signals 
                    WHERE symbol = ? AND signal_type = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, signal_type, limit))
            else:
                cursor = await conn.execute("""
                    SELECT symbol, timestamp, signal_type, direction, strength, confidence, source
                    FROM trading_signals 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, limit))
                
            rows = await cursor.fetchall()
            await self._release_connection(conn)
            
            data = []
            for row in rows:
                data.append({
                    'symbol': row[0],
                    'timestamp': row[1],
                    'signal_type': row[2],
                    'direction': row[3],
                    'strength': row[4],
                    'confidence': row[5],
                    'source': row[6]
                })
                
            return data
            
        except Exception as e:
            logger.error(f"Failed to get trading signals: {e}")
            return []
            
    async def store_metrics(self, metrics: Dict):
        """Store system metrics"""
        try:
            conn = await self._get_connection()
            
            timestamp = metrics.get('timestamp', time.time())
            
            for metric_name, value in metrics.items():
                if metric_name != 'timestamp' and metric_name != 'module_status':
                    if isinstance(value, dict):
                        value = json.dumps(value)
                    elif not isinstance(value, (int, float)):
                        continue
                        
                    await conn.execute("""
                        INSERT INTO system_metrics (timestamp, metric_name, value)
                        VALUES (?, ?, ?)
                    """, (timestamp, metric_name, value))
                    
            await conn.commit()
            await self._release_connection(conn)
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
            
    async def store_alert(self, message: str, level: str, source: str = "system"):
        """Store system alert"""
        try:
            conn = await self._get_connection()
            
            await conn.execute("""
                INSERT INTO alerts (timestamp, level, message, source)
                VALUES (?, ?, ?, ?)
            """, (time.time(), level, message, source))
            
            await conn.commit()
            await self._release_connection(conn)
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            
    def _update_cache(self, key: str, data: Any):
        """Update cache with new data"""
        try:
            # Serialize data for size calculation
            data_size = len(pickle.dumps(data))
            
            # Check if adding this data would exceed cache size
            if self.cache_size + data_size > self.max_cache_size:
                # Remove oldest entries
                self._cleanup_cache()
                
            # Add to cache
            self.cache[key] = data
            self.cache_size += data_size
            
        except Exception as e:
            logger.error(f"Cache update failed: {e}")
            
    def _cleanup_cache(self):
        """Clean up cache to free memory"""
        try:
            # Remove oldest entries (simple FIFO for now)
            while self.cache_size > self.max_cache_size * 0.8 and self.cache:
                key = next(iter(self.cache))
                data = self.cache.pop(key)
                self.cache_size -= len(pickle.dumps(data))
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            
    async def backup_database(self, backup_path: str = None):
        """Create database backup"""
        try:
            if backup_path is None:
                backup_dir = Path("backup")
                backup_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"mdps_backup_{timestamp}.db"
                
            # Create backup
            conn = await self._get_connection()
            await conn.execute("VACUUM INTO ?", (str(backup_path),))
            await self._release_connection(conn)
            
            # Compress backup
            with open(backup_path, 'rb') as f_in:
                with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
                    
            # Remove uncompressed backup
            backup_path.unlink()
            
            logger.info(f"Database backup created: {backup_path}.gz")
            return f"{backup_path}.gz"
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
            
    async def restore_database(self, backup_path: str):
        """Restore database from backup"""
        try:
            # Decompress backup
            with gzip.open(backup_path, 'rb') as f_in:
                temp_path = backup_path.replace('.gz', '_temp')
                with open(temp_path, 'wb') as f_out:
                    f_out.writelines(f_in)
                    
            # Close all connections
            await self.disconnect()
            
            # Replace database file
            import shutil
            shutil.copy2(temp_path, self.db_path)
            os.remove(temp_path)
            
            # Reconnect
            await self.connect()
            
            logger.info(f"Database restored from: {backup_path}")
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            raise
            
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            conn = await self._get_connection()
            
            stats = {}
            
            # Table row counts
            tables = ['market_data', 'technical_indicators', 'ml_predictions', 
                     'trading_signals', 'system_metrics', 'alerts']
                     
            for table in tables:
                cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = await cursor.fetchone()
                stats[f"{table}_count"] = count[0] if count else 0
                
            # Database size
            cursor = await conn.execute("PRAGMA page_count")
            page_count = await cursor.fetchone()
            cursor = await conn.execute("PRAGMA page_size")
            page_size = await cursor.fetchone()
            
            if page_count and page_size:
                stats['database_size_bytes'] = page_count[0] * page_size[0]
                
            await self._release_connection(conn)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

# Global database instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

async def initialize_database():
    """Initialize global database manager"""
    global db_manager
    db_manager = DatabaseManager()
    await db_manager.connect()
    return db_manager

if __name__ == "__main__":
    # Test database functionality
    async def test_database():
        db = DatabaseManager()
        await db.connect()
        
        # Test market data storage
        await db.store_market_data(
            symbol="BTCUSD",
            timestamp=time.time(),
            open_price=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            timeframe="1h"
        )
        
        # Test data retrieval
        data = await db.get_market_data("BTCUSD", "1h", limit=10)
        print(f"Retrieved {len(data)} market data records")
        
        # Test database stats
        stats = await db.get_database_stats()
        print("Database stats:", stats)
        
        await db.disconnect()
        
    asyncio.run(test_database())