# candle_constructor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
import logging

class CandleConstructor:
    """
    A class to aggregate tick or second-level data into OHLCV bar charts.
    Handles different bar types (time, volume, tick, range, etc.).
    """
    
    def __init__(self, buffer_size=10000):
        """
        Initialize Candle Constructor
        
        Args:
            buffer_size (int): Memory buffer size
        """
        self.buffer_size = buffer_size
        self.candle_buffer = queue.Queue(maxsize=buffer_size)
        self.logger = self._setup_logger()
        
        # Construction status
        self.is_constructing = False
        self.construction_thread = None
        
        # Bar types
        self.BAR_TYPE_TIME = "time"
        self.BAR_TYPE_VOLUME = "volume"
        self.BAR_TYPE_TICK = "tick"
        self.BAR_TYPE_RANGE = "range"
        self.BAR_TYPE_RENKO = "renko"
        self.BAR_TYPE_HEIKIN_ASHI = "heikin_ashi"
        
        # Active constructions
        self.active_constructions = {}
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("CandleConstructor")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("candle_constructor.log")
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
    
    def start_construction(self, construction_id, bar_type, bar_size, data_source, callback=None):
        """
        Start candle construction
        
        Args:
            construction_id (str): Unique construction identifier
            bar_type (str): Type of bars to construct
            bar_size (float): Size of bars (time in seconds, volume in lots, etc.)
            data_source (function): Function to get data (returns list of ticks or candles)
            callback (function): Callback function to handle constructed candles
            
        Returns:
            bool: Whether successfully started construction
        """
        if construction_id in self.active_constructions:
            self.logger.warning(f"Candle construction {construction_id} is already running")
            return False
        
        # Validate bar type
        valid_bar_types = [
            self.BAR_TYPE_TIME,
            self.BAR_TYPE_VOLUME,
            self.BAR_TYPE_TICK,
            self.BAR_TYPE_RANGE,
            self.BAR_TYPE_RENKO,
            self.BAR_TYPE_HEIKIN_ASHI
        ]
        
        if bar_type not in valid_bar_types:
            self.logger.error(f"Invalid bar type: {bar_type}")
            return False
        
        # Create construction context
        context = {
            'id': construction_id,
            'bar_type': bar_type,
            'bar_size': bar_size,
            'data_source': data_source,
            'callback': callback,
            'last_candle': None,
            'pending_ticks': [],
            'pending_volume': 0,
            'pending_tick_count': 0,
            'pending_high': None,
            'pending_low': None,
            'pending_open': None,
            'pending_close': None,
            'pending_time': None
        }
        
        # Add to active constructions
        self.active_constructions[construction_id] = context
        
        # If this is the first construction, start the construction thread
        if not self.is_constructing:
            self.is_constructing = True
            self.construction_thread = threading.Thread(target=self._construct_candles)
            self.construction_thread.daemon = True
            self.construction_thread.start()
        
        self.logger.info(f"Started candle construction {construction_id}: {bar_type} bars of size {bar_size}")
        return True
    
    def stop_construction(self, construction_id):
        """
        Stop candle construction
        
        Args:
            construction_id (str): Construction identifier
            
        Returns:
            bool: Whether successfully stopped construction
        """
        if construction_id not in self.active_constructions:
            self.logger.warning(f"Candle construction {construction_id} not found")
            return False
        
        # Remove from active constructions
        del self.active_constructions[construction_id]
        
        # If no more active constructions, stop the construction thread
        if not self.active_constructions and self.is_constructing:
            self.is_constructing = False
            if self.construction_thread and self.construction_thread.is_alive():
                self.construction_thread.join(timeout=5)
        
        self.logger.info(f"Stopped candle construction {construction_id}")
        return True
    
    def _construct_candles(self):
        """
        Internal method to construct candles, runs in separate thread
        """
        while self.is_constructing:
            try:
                # Process each active construction
                for construction_id, context in list(self.active_constructions.items()):
                    # Get data from source
                    data = context['data_source']()
                    
                    if data is None or not data:
                        continue
                    
                    # Process based on bar type
                    if context['bar_type'] == self.BAR_TYPE_TIME:
                        self._construct_time_bars(context, data)
                    elif context['bar_type'] == self.BAR_TYPE_VOLUME:
                        self._construct_volume_bars(context, data)
                    elif context['bar_type'] == self.BAR_TYPE_TICK:
                        self._construct_tick_bars(context, data)
                    elif context['bar_type'] == self.BAR_TYPE_RANGE:
                        self._construct_range_bars(context, data)
                    elif context['bar_type'] == self.BAR_TYPE_RENKO:
                        self._construct_renko_bars(context, data)
                    elif context['bar_type'] == self.BAR_TYPE_HEIKIN_ASHI:
                        self._construct_heikin_ashi_bars(context, data)
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error constructing candles: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def _construct_time_bars(self, context, data):
        """
        Construct time-based bars
        
        Args:
            context (dict): Construction context
            data (list): List of ticks or candles
        """
        bar_size = context['bar_size']
        
        for item in data:
            # Extract timestamp and price
            if isinstance(item, dict):
                timestamp = item.get('time') or item.get('time_msc')
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.timestamp()
                
                price = item.get('last') or item.get('close')
                volume = item.get('volume') or item.get('tick_volume') or 1
            else:
                # Assume it's a pandas Series or similar
                timestamp = item.get('time', item.get('time_msc'))
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.timestamp()
                
                price = item.get('last', item.get('close'))
                volume = item.get('volume', item.get('tick_volume', 1))
            
            if timestamp is None or price is None:
                continue
            
            # Calculate bar start time
            bar_start_time = int(timestamp / bar_size) * bar_size
            
            # Initialize pending values if needed
            if context['pending_time'] is None:
                context['pending_time'] = bar_start_time
                context['pending_open'] = price
                context['pending_high'] = price
                context['pending_low'] = price
                context['pending_close'] = price
                context['pending_volume'] = volume
                context['pending_tick_count'] = 1
            elif bar_start_time == context['pending_time']:
                # Update pending values
                context['pending_high'] = max(context['pending_high'], price)
                context['pending_low'] = min(context['pending_low'], price)
                context['pending_close'] = price
                context['pending_volume'] += volume
                context['pending_tick_count'] += 1
            else:
                # Bar is complete, create candle
                candle = {
                    'time': datetime.fromtimestamp(context['pending_time']),
                    'open': context['pending_open'],
                    'high': context['pending_high'],
                    'low': context['pending_low'],
                    'close': context['pending_close'],
                    'volume': context['pending_volume'],
                    'tick_volume': context['pending_tick_count']
                }
                
                # Add candle to buffer
                if self.candle_buffer.full():
                    # If buffer is full, remove oldest candle
                    try:
                        self.candle_buffer.get_nowait()
                    except queue.Empty:
                        pass
                
                self.candle_buffer.put(candle)
                
                # Call callback if provided
                if context['callback']:
                    try:
                        context['callback'](candle)
                    except Exception as e:
                        self.logger.error(f"Error in candle callback: {str(e)}")
                
                # Start new bar
                context['pending_time'] = bar_start_time
                context['pending_open'] = price
                context['pending_high'] = price
                context['pending_low'] = price
                context['pending_close'] = price
                context['pending_volume'] = volume
                context['pending_tick_count'] = 1
            
            # Store tick for potential use in other bar types
            context['pending_ticks'].append(item)
    
    def _construct_volume_bars(self, context, data):
        """
        Construct volume-based bars
        
        Args:
            context (dict): Construction context
            data (list): List of ticks or candles
        """
        bar_size = context['bar_size']
        
        for item in data:
            # Extract price and volume
            if isinstance(item, dict):
                price = item.get('last') or item.get('close')
                volume = item.get('volume') or item.get('tick_volume') or 1
            else:
                # Assume it's a pandas Series or similar
                price = item.get('last', item.get('close'))
                volume = item.get('volume', item.get('tick_volume', 1))
            
            if price is None:
                continue
            
            # Initialize pending values if needed
            if context['pending_open'] is None:
                context['pending_open'] = price
                context['pending_high'] = price
                context['pending_low'] = price
                context['pending_close'] = price
                context['pending_volume'] = volume
                context['pending_tick_count'] = 1
                context['pending_time'] = time.time()
            else:
                # Update pending values
                context['pending_high'] = max(context['pending_high'], price)
                context['pending_low'] = min(context['pending_low'], price)
                context['pending_close'] = price
                context['pending_volume'] += volume
                context['pending_tick_count'] += 1
                
                # Check if bar is complete
                if context['pending_volume'] >= bar_size:
                    # Create candle
                    candle = {
                        'time': datetime.fromtimestamp(context['pending_time']),
                        'open': context['pending_open'],
                        'high': context['pending_high'],
                        'low': context['pending_low'],
                        'close': context['pending_close'],
                        'volume': context['pending_volume'],
                        'tick_volume': context['pending_tick_count']
                    }
                    
                    # Add candle to buffer
                    if self.candle_buffer.full():
                        # If buffer is full, remove oldest candle
                        try:
                            self.candle_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.candle_buffer.put(candle)
                    
                    # Call callback if provided
                    if context['callback']:
                        try:
                            context['callback'](candle)
                        except Exception as e:
                            self.logger.error(f"Error in candle callback: {str(e)}")
                    
                    # Reset pending values
                    context['pending_open'] = price
                    context['pending_high'] = price
                    context['pending_low'] = price
                    context['pending_close'] = price
                    context['pending_volume'] = 0
                    context['pending_tick_count'] = 0
                    context['pending_time'] = time.time()
            
            # Store tick for potential use in other bar types
            context['pending_ticks'].append(item)
    
    def _construct_tick_bars(self, context, data):
        """
        Construct tick-based bars
        
        Args:
            context (dict): Construction context
            data (list): List of ticks or candles
        """
        bar_size = context['bar_size']
        
        for item in data:
            # Extract price and volume
            if isinstance(item, dict):
                price = item.get('last') or item.get('close')
                volume = item.get('volume') or item.get('tick_volume') or 1
            else:
                # Assume it's a pandas Series or similar
                price = item.get('last', item.get('close'))
                volume = item.get('volume', item.get('tick_volume', 1))
            
            if price is None:
                continue
            
            # Initialize pending values if needed
            if context['pending_open'] is None:
                context['pending_open'] = price
                context['pending_high'] = price
                context['pending_low'] = price
                context['pending_close'] = price
                context['pending_volume'] = volume
                context['pending_tick_count'] = 1
                context['pending_time'] = time.time()
            else:
                # Update pending values
                context['pending_high'] = max(context['pending_high'], price)
                context['pending_low'] = min(context['pending_low'], price)
                context['pending_close'] = price
                context['pending_volume'] += volume
                context['pending_tick_count'] += 1
                
                # Check if bar is complete
                if context['pending_tick_count'] >= bar_size:
                    # Create candle
                    candle = {
                        'time': datetime.fromtimestamp(context['pending_time']),
                        'open': context['pending_open'],
                        'high': context['pending_high'],
                        'low': context['pending_low'],
                        'close': context['pending_close'],
                        'volume': context['pending_volume'],
                        'tick_volume': context['pending_tick_count']
                    }
                    
                    # Add candle to buffer
                    if self.candle_buffer.full():
                        # If buffer is full, remove oldest candle
                        try:
                            self.candle_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.candle_buffer.put(candle)
                    
                    # Call callback if provided
                    if context['callback']:
                        try:
                            context['callback'](candle)
                        except Exception as e:
                            self.logger.error(f"Error in candle callback: {str(e)}")
                    
                    # Reset pending values
                    context['pending_open'] = price
                    context['pending_high'] = price
                    context['pending_low'] = price
                    context['pending_close'] = price
                    context['pending_volume'] = 0
                    context['pending_tick_count'] = 0
                    context['pending_time'] = time.time()
            
            # Store tick for potential use in other bar types
            context['pending_ticks'].append(item)
    
    def _construct_range_bars(self, context, data):
        """
        Construct range-based bars
        
        Args:
            context (dict): Construction context
            data (list): List of ticks or candles
        """
        bar_size = context['bar_size']
        
        for item in data:
            # Extract price and volume
            if isinstance(item, dict):
                price = item.get('last') or item.get('close')
                volume = item.get('volume') or item.get('tick_volume') or 1
            else:
                # Assume it's a pandas Series or similar
                price = item.get('last', item.get('close'))
                volume = item.get('volume', item.get('tick_volume', 1))
            
            if price is None:
                continue
            
            # Initialize pending values if needed
            if context['pending_open'] is None:
                context['pending_open'] = price
                context['pending_high'] = price
                context['pending_low'] = price
                context['pending_close'] = price
                context['pending_volume'] = volume
                context['pending_tick_count'] = 1
                context['pending_time'] = time.time()
            else:
                # Update pending values
                context['pending_high'] = max(context['pending_high'], price)
                context['pending_low'] = min(context['pending_low'], price)
                context['pending_close'] = price
                context['pending_volume'] += volume
                context['pending_tick_count'] += 1
                
                # Check if bar is complete
                if context['pending_high'] - context['pending_low'] >= bar_size:
                    # Create candle
                    candle = {
                        'time': datetime.fromtimestamp(context['pending_time']),
                        'open': context['pending_open'],
                        'high': context['pending_high'],
                        'low': context['pending_low'],
                        'close': context['pending_close'],
                        'volume': context['pending_volume'],
                        'tick_volume': context['pending_tick_count']
                    }
                    
                    # Add candle to buffer
                    if self.candle_buffer.full():
                        # If buffer is full, remove oldest candle
                        try:
                            self.candle_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.candle_buffer.put(candle)
                    
                    # Call callback if provided
                    if context['callback']:
                        try:
                            context['callback'](candle)
                        except Exception as e:
                            self.logger.error(f"Error in candle callback: {str(e)}")
                    
                    # Reset pending values
                    # For range bars, the open of the new bar is the close of the previous bar
                    context['pending_open'] = price
                    context['pending_high'] = price
                    context['pending_low'] = price
                    context['pending_close'] = price
                    context['pending_volume'] = 0
                    context['pending_tick_count'] = 0
                    context['pending_time'] = time.time()
            
            # Store tick for potential use in other bar types
            context['pending_ticks'].append(item)
    
    def _construct_renko_bars(self, context, data):
        """
        Construct Renko bars
        
        Args:
            context (dict): Construction context
            data (list): List of ticks or candles
        """
        bar_size = context['bar_size']
        
        for item in data:
            # Extract price and volume
            if isinstance(item, dict):
                price = item.get('last') or item.get('close')
                volume = item.get('volume') or item.get('tick_volume') or 1
            else:
                # Assume it's a pandas Series or similar
                price = item.get('last', item.get('close'))
                volume = item.get('volume', item.get('tick_volume', 1))
            
            if price is None:
                continue
            
            # Initialize pending values if needed
            if context['pending_open'] is None:
                context['pending_open'] = price
                context['pending_high'] = price
                context['pending_low'] = price
                context['pending_close'] = price
                context['pending_volume'] = volume
                context['pending_tick_count'] = 1
                context['pending_time'] = time.time()
            else:
                # Update pending values
                context['pending_high'] = max(context['pending_high'], price)
                context['pending_low'] = min(context['pending_low'], price)
                context['pending_close'] = price
                context['pending_volume'] += volume
                context['pending_tick_count'] += 1
                
                # Check if bar is complete
                # For Renko bars, we need to check if price has moved by at least bar_size
                if price >= context['pending_open'] + bar_size:
                    # Up bar
                    # Create candle
                    candle = {
                        'time': datetime.fromtimestamp(context['pending_time']),
                        'open': context['pending_open'],
                        'high': context['pending_open'] + bar_size,
                        'low': context['pending_open'],
                        'close': context['pending_open'] + bar_size,
                        'volume': context['pending_volume'],
                        'tick_volume': context['pending_tick_count']
                    }
                    
                    # Add candle to buffer
                    if self.candle_buffer.full():
                        # If buffer is full, remove oldest candle
                        try:
                            self.candle_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.candle_buffer.put(candle)
                    
                    # Call callback if provided
                    if context['callback']:
                        try:
                            context['callback'](candle)
                        except Exception as e:
                            self.logger.error(f"Error in candle callback: {str(e)}")
                    
                    # Reset pending values
                    context['pending_open'] = context['pending_open'] + bar_size
                    context['pending_high'] = context['pending_open']
                    context['pending_low'] = context['pending_open']
                    context['pending_close'] = context['pending_open']
                    context['pending_volume'] = 0
                    context['pending_tick_count'] = 0
                    context['pending_time'] = time.time()
                    
                    # Check if we need to create additional bars
                    while price >= context['pending_open'] + bar_size:
                        # Create another up bar
                        candle = {
                            'time': datetime.fromtimestamp(context['pending_time']),
                            'open': context['pending_open'],
                            'high': context['pending_open'] + bar_size,
                            'low': context['pending_open'],
                            'close': context['pending_open'] + bar_size,
                            'volume': 0,
                            'tick_volume': 0
                        }
                        
                        # Add candle to buffer
                        if self.candle_buffer.full():
                            # If buffer is full, remove oldest candle
                            try:
                                self.candle_buffer.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.candle_buffer.put(candle)
                        
                        # Call callback if provided
                        if context['callback']:
                            try:
                                context['callback'](candle)
                            except Exception as e:
                                self.logger.error(f"Error in candle callback: {str(e)}")
                        
                        # Update pending values
                        context['pending_open'] = context['pending_open'] + bar_size
                        context['pending_high'] = context['pending_open']
                        context['pending_low'] = context['pending_open']
                        context['pending_close'] = context['pending_open']
                        context['pending_time'] = time.time()
                
                elif price <= context['pending_open'] - bar_size:
                    # Down bar
                    # Create candle
                    candle = {
                        'time': datetime.fromtimestamp(context['pending_time']),
                        'open': context['pending_open'],
                        'high': context['pending_open'],
                        'low': context['pending_open'] - bar_size,
                        'close': context['pending_open'] - bar_size,
                        'volume': context['pending_volume'],
                        'tick_volume': context['pending_tick_count']
                    }
                    
                    # Add candle to buffer
                    if self.candle_buffer.full():
                        # If buffer is full, remove oldest candle
                        try:
                            self.candle_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.candle_buffer.put(candle)
                    
                    # Call callback if provided
                    if context['callback']:
                        try:
                            context['callback'](candle)
                        except Exception as e:
                            self.logger.error(f"Error in candle callback: {str(e)}")
                    
                    # Reset pending values
                    context['pending_open'] = context['pending_open'] - bar_size
                    context['pending_high'] = context['pending_open']
                    context['pending_low'] = context['pending_open']
                    context['pending_close'] = context['pending_open']
                    context['pending_volume'] = 0
                    context['pending_tick_count'] = 0
                    context['pending_time'] = time.time()
                    
                    # Check if we need to create additional bars
                    while price <= context['pending_open'] - bar_size:
                        # Create another down bar
                        candle = {
                            'time': datetime.fromtimestamp(context['pending_time']),
                            'open': context['pending_open'],
                            'high': context['pending_open'],
                            'low': context['pending_open'] - bar_size,
                            'close': context['pending_open'] - bar_size,
                            'volume': 0,
                            'tick_volume': 0
                        }
                        
                        # Add candle to buffer
                        if self.candle_buffer.full():
                            # If buffer is full, remove oldest candle
                            try:
                                self.candle_buffer.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.candle_buffer.put(candle)
                        
                        # Call callback if provided
                        if context['callback']:
                            try:
                                context['callback'](candle)
                            except Exception as e:
                                self.logger.error(f"Error in candle callback: {str(e)}")
                        
                        # Update pending values
                        context['pending_open'] = context['pending_open'] - bar_size
                        context['pending_high'] = context['pending_open']
                        context['pending_low'] = context['pending_open']
                        context['pending_close'] = context['pending_open']
                        context['pending_time'] = time.time()
            
            # Store tick for potential use in other bar types
            context['pending_ticks'].append(item)
    
    def _construct_heikin_ashi_bars(self, context, data):
        """
        Construct Heikin-Ashi bars
        
        Args:
            context (dict): Construction context
            data (list): List of ticks or candles
        """
        bar_size = context['bar_size']
        
        for item in data:
            # Extract timestamp and price
            if isinstance(item, dict):
                timestamp = item.get('time') or item.get('time_msc')
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.timestamp()
                
                price = item.get('last') or item.get('close')
                volume = item.get('volume') or item.get('tick_volume') or 1
            else:
                # Assume it's a pandas Series or similar
                timestamp = item.get('time', item.get('time_msc'))
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.timestamp()
                
                price = item.get('last', item.get('close'))
                volume = item.get('volume', item.get('tick_volume', 1))
            
            if timestamp is None or price is None:
                continue
            
            # Calculate bar start time
            bar_start_time = int(timestamp / bar_size) * bar_size
            
            # Initialize pending values if needed
            if context['pending_time'] is None:
                context['pending_time'] = bar_start_time
                context['pending_open'] = price
                context['pending_high'] = price
                context['pending_low'] = price
                context['pending_close'] = price
                context['pending_volume'] = volume
                context['pending_tick_count'] = 1
            elif bar_start_time == context['pending_time']:
                # Update pending values
                context['pending_high'] = max(context['pending_high'], price)
                context['pending_low'] = min(context['pending_low'], price)
                context['pending_close'] = price
                context['pending_volume'] += volume
                context['pending_tick_count'] += 1
            else:
                # Bar is complete, create Heikin-Ashi candle
                # Heikin-Ashi calculations:
                # HA_Close = (Open + High + Low + Close) / 4
                # HA_Open = (previous HA_Open + previous HA_Close) / 2
                # HA_High = max(High, HA_Open, HA_Close)
                # HA_Low = min(Low, HA_Open, HA_Close)
                
                if context['last_candle'] is None:
                    # First candle, use regular values
                    ha_open = (context['pending_open'] + context['pending_close']) / 2
                    ha_close = (context['pending_open'] + context['pending_high'] + context['pending_low'] + context['pending_close']) / 4
                    ha_high = max(context['pending_high'], ha_open, ha_close)
                    ha_low = min(context['pending_low'], ha_open, ha_close)
                else:
                    # Use previous Heikin-Ashi values
                    ha_open = (context['last_candle']['ha_open'] + context['last_candle']['ha_close']) / 2
                    ha_close = (context['pending_open'] + context['pending_high'] + context['pending_low'] + context['pending_close']) / 4
                    ha_high = max(context['pending_high'], ha_open, ha_close)
                    ha_low = min(context['pending_low'], ha_open, ha_close)
                
                # Create candle
                candle = {
                    'time': datetime.fromtimestamp(context['pending_time']),
                    'open': context['pending_open'],
                    'high': context['pending_high'],
                    'low': context['pending_low'],
                    'close': context['pending_close'],
                    'volume': context['pending_volume'],
                    'tick_volume': context['pending_tick_count'],
                    'ha_open': ha_open,
                    'ha_high': ha_high,
                    'ha_low': ha_low,
                    'ha_close': ha_close
                }
                
                # Add candle to buffer
                if self.candle_buffer.full():
                    # If buffer is full, remove oldest candle
                    try:
                        self.candle_buffer.get_nowait()
                    except queue.Empty:
                        pass
                
                self.candle_buffer.put(candle)
                
                # Store as last candle for next Heikin-Ashi calculation
                context['last_candle'] = {
                    'ha_open': ha_open,
                    'ha_high': ha_high,
                    'ha_low': ha_low,
                    'ha_close': ha_close
                }
                
                # Call callback if provided
                if context['callback']:
                    try:
                        context['callback'](candle)
                    except Exception as e:
                        self.logger.error(f"Error in candle callback: {str(e)}")
                
                # Start new bar
                context['pending_time'] = bar_start_time
                context['pending_open'] = price
                context['pending_high'] = price
                context['pending_low'] = price
                context['pending_close'] = price
                context['pending_volume'] = volume
                context['pending_tick_count'] = 1
            
            # Store tick for potential use in other bar types
            context['pending_ticks'].append(item)
    
    def get_candles(self, count=None):
        """
        Get candles from buffer
        
        Args:
            count (int): Number of candles to get, None means get all
            
        Returns:
            list: List of candles
        """
        candles = []
        
        if count is None:
            # Get all candles
            while not self.candle_buffer.empty():
                try:
                    candles.append(self.candle_buffer.get_nowait())
                except queue.Empty:
                    break
        else:
            # Get specified number of candles
            for _ in range(min(count, self.candle_buffer.qsize())):
                try:
                    candles.append(self.candle_buffer.get_nowait())
                except queue.Empty:
                    break
        
        return candles
    
    def get_candles_dataframe(self, count=None):
        """
        Get candles from buffer and convert to DataFrame
        
        Args:
            count (int): Number of candles to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing candles
        """
        candles = self.get_candles(count)
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        
        # Convert timestamps
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    def save_candles_to_csv(self, filename, count=None):
        """
        Save candles in buffer to CSV file
        
        Args:
            filename (str): Filename
            count (int): Number of candles to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_candles_dataframe(count)
            
            if df.empty:
                self.logger.warning("No candles to save")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} candles to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save candles to CSV: {str(e)}")
            return False
    
    def save_candles_to_parquet(self, filename, count=None):
        """
        Save candles in buffer to Parquet file
        
        Args:
            filename (str): Filename
            count (int): Number of candles to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_candles_dataframe(count)
            
            if df.empty:
                self.logger.warning("No candles to save")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} candles to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save candles to Parquet: {str(e)}")
            return False
