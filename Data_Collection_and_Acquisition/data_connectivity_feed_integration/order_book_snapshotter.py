# order_book_snapshotter.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import threading
import queue
import logging
import os

class OrderBookSnapshotter:
    """
    A class to periodically capture Level 1/Level 2 order book snapshots.
    Tracks order book depth, liquidity, and imbalance metrics.
    """
    
    def __init__(self, mt5_connector, buffer_size=10000):
        """
        Initialize Order Book Snapshotter
        
        Args:
            mt5_connector (MetaTrader5Connector): MT5 connector instance
            buffer_size (int): Memory buffer size
        """
        self.mt5_connector = mt5_connector
        self.buffer_size = buffer_size
        self.snapshot_buffer = queue.Queue(maxsize=buffer_size)
        self.is_snapshotting = False
        self.snapshot_thread = None
        self.logger = self._setup_logger()
        self.snapshot_interval = 5  # seconds
        self.order_book_history = {}
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("OrderBookSnapshotter")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("order_book_snapshotter.log")
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
    
    def set_snapshot_interval(self, interval):
        """
        Set snapshot interval
        
        Args:
            interval (int): Snapshot interval in seconds
        """
        self.snapshot_interval = interval
        self.logger.info(f"Set snapshot interval to {interval} seconds")
    
    def start_snapshotting(self, symbol):
        """
        Start taking order book snapshots for specified symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: Whether successfully started snapshotting
        """
        if self.is_snapshotting:
            self.logger.warning("Order book snapshotting is already running")
            return False
        
        # Ensure MT5 is connected
        if not self.mt5_connector.connected:
            if not self.mt5_connector.connect():
                self.logger.error("Failed to connect to MetaTrader 5")
                return False
        
        # Initialize history for symbol
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = []
        
        # Start snapshotting thread
        self.is_snapshotting = True
        self.snapshot_thread = threading.Thread(
            target=self._take_snapshots, 
            args=(symbol,)
        )
        self.snapshot_thread.daemon = True
        self.snapshot_thread.start()
        
        self.logger.info(f"Started order book snapshotting for {symbol}")
        return True
    
    def stop_snapshotting(self):
        """
        Stop taking order book snapshots
        
        Returns:
            bool: Whether successfully stopped snapshotting
        """
        if not self.is_snapshotting:
            self.logger.warning("Order book snapshotting is not running")
            return False
        
        # Stop snapshotting thread
        self.is_snapshotting = False
        if self.snapshot_thread and self.snapshot_thread.is_alive():
            self.snapshot_thread.join(timeout=5)
        
        self.logger.info("Stopped order book snapshotting")
        return True
    
    def _take_snapshots(self, symbol):
        """
        Internal method to take order book snapshots, runs in separate thread
        
        Args:
            symbol (str): Trading symbol
        """
        while self.is_snapshotting:
            try:
                # Get current time
                current_time = datetime.now()
                
                # Get order book
                order_book = mt5.market_book_get(symbol)
                
                if order_book is not None and len(order_book) > 0:
                    # Convert to DataFrame for easier processing
                    df = pd.DataFrame(list(order_book), columns=order_book[0]._asdict().keys())
                    
                    # Calculate metrics
                    bid_volume = df[df['type'] == 1]['volume'].sum()  # Sell orders (type 1)
                    ask_volume = df[df['type'] == 0]['volume'].sum()  # Buy orders (type 0)
                    
                    # Calculate best bid/ask
                    best_bid = df[df['type'] == 1]['price'].max() if len(df[df['type'] == 1]) > 0 else None
                    best_ask = df[df['type'] == 0]['price'].min() if len(df[df['type'] == 0]) > 0 else None
                    
                    # Calculate spread
                    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
                    
                    # Calculate imbalance
                    total_volume = bid_volume + ask_volume
                    imbalance = (ask_volume - bid_volume) / total_volume if total_volume > 0 else 0
                    
                    # Create snapshot data dictionary
                    snapshot_data = {
                        'time': current_time,
                        'symbol': symbol,
                        'bid_volume': bid_volume,
                        'ask_volume': ask_volume,
                        'total_volume': total_volume,
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'spread': spread,
                        'imbalance': imbalance,
                        'levels': len(df)
                    }
                    
                    # Add order book levels
                    for i, (_, row) in enumerate(df.iterrows()):
                        snapshot_data[f'level_{i}_type'] = row['type']
                        snapshot_data[f'level_{i}_price'] = row['price']
                        snapshot_data[f'level_{i}_volume'] = row['volume']
                    
                    # Put snapshot data into buffer
                    if self.snapshot_buffer.full():
                        # If buffer is full, remove oldest data
                        try:
                            self.snapshot_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.snapshot_buffer.put(snapshot_data)
                    
                    # Update history
                    self.order_book_history[symbol].append({
                        'time': current_time,
                        'bid_volume': bid_volume,
                        'ask_volume': ask_volume,
                        'total_volume': total_volume,
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'spread': spread,
                        'imbalance': imbalance,
                        'levels': len(df)
                    })
                    
                    # Keep only recent history (last 1000 entries)
                    if len(self.order_book_history[symbol]) > 1000:
                        self.order_book_history[symbol].pop(0)
                    
                    # Log
                    self.logger.debug(f"Captured order book snapshot for {symbol}: bid={best_bid}, ask={best_ask}, spread={spread}, imbalance={imbalance:.2f}")
                else:
                    self.logger.warning(f"No order book data for {symbol}")
                
                # Sleep until next snapshot
                time.sleep(self.snapshot_interval)
                
            except Exception as e:
                self.logger.error(f"Error taking order book snapshot: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def get_snapshots(self, count=None):
        """
        Get order book snapshots from buffer
        
        Args:
            count (int): Number of snapshots to get, None means get all
            
        Returns:
            list: List of order book snapshots
        """
        snapshots = []
        
        if count is None:
            # Get all snapshots
            while not self.snapshot_buffer.empty():
                try:
                    snapshots.append(self.snapshot_buffer.get_nowait())
                except queue.Empty:
                    break
        else:
            # Get specified number of snapshots
            for _ in range(min(count, self.snapshot_buffer.qsize())):
                try:
                    snapshots.append(self.snapshot_buffer.get_nowait())
                except queue.Empty:
                    break
        
        return snapshots
    
    def get_snapshots_dataframe(self, count=None):
        """
        Get order book snapshots from buffer and convert to DataFrame
        
        Args:
            count (int): Number of snapshots to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing order book snapshots
        """
        snapshots = self.get_snapshots(count)
        
        if not snapshots:
            return pd.DataFrame()
        
        df = pd.DataFrame(snapshots)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    def get_latest_snapshot(self, symbol):
        """
        Get latest order book snapshot for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Latest snapshot or None if not available
        """
        if symbol not in self.order_book_history or not self.order_book_history[symbol]:
            return None
        
        return self.order_book_history[symbol][-1]
    
    def get_order_book_stats(self, symbol):
        """
        Get order book statistics for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Order book statistics
        """
        if symbol not in self.order_book_history or not self.order_book_history[symbol]:
            return {}
        
        history = self.order_book_history[symbol]
        
        # Extract metrics
        spreads = [entry['spread'] for entry in history if entry['spread'] is not None]
        imbalances = [entry['imbalance'] for entry in history]
        total_volumes = [entry['total_volume'] for entry in history]
        levels = [entry['levels'] for entry in history]
        
        return {
            'avg_spread': sum(spreads) / len(spreads) if spreads else None,
            'min_spread': min(spreads) if spreads else None,
            'max_spread': max(spreads) if spreads else None,
            'avg_imbalance': sum(imbalances) / len(imbalances) if imbalances else None,
            'avg_total_volume': sum(total_volumes) / len(total_volumes) if total_volumes else None,
            'avg_levels': sum(levels) / len(levels) if levels else None,
            'samples': len(history)
        }
    
    def save_snapshots_to_csv(self, filename, count=None):
        """
        Save order book snapshots in buffer to CSV file
        
        Args:
            filename (str): Filename
            count (int): Number of snapshots to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_snapshots_dataframe(count)
            
            if df.empty:
                self.logger.warning("No order book snapshots to save")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} order book snapshots to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save order book snapshots to CSV: {str(e)}")
            return False
    
    def save_snapshots_to_parquet(self, filename, count=None):
        """
        Save order book snapshots in buffer to Parquet file
        
        Args:
            filename (str): Filename
            count (int): Number of snapshots to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_snapshots_dataframe(count)
            
            if df.empty:
                self.logger.warning("No order book snapshots to save")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} order book snapshots to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save order book snapshots to Parquet: {str(e)}")
            return False
