import sqlite3
from typing import List, Dict, Any
import json

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    pnl REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            """)
            
            conn.commit()

    def insert_market_data(self, data: Dict[str, Any]):
        """Insert market data into database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO market_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data['symbol'],
                data['timestamp'],
                data['open'],
                data['high'],
                data['low'],
                data['close'],
                data['volume']
            ))
            conn.commit()

    def insert_trade(self, trade: Dict[str, Any]):
        """Insert trade into database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades 
                (symbol, timestamp, side, price, quantity, pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                trade['symbol'],
                trade['timestamp'],
                trade['side'],
                trade['price'],
                trade['quantity'],
                trade['pnl']
            ))
            conn.commit()

    def insert_order(self, order: Dict[str, Any]):
        """Insert order into database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO orders 
                (order_id, symbol, type, side, quantity, price, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order['order_id'],
                order['symbol'],
                order['type'],
                order['side'],
                order['quantity'],
                order['price'],
                order['status'],
                order['timestamp']
            ))
            conn.commit()
