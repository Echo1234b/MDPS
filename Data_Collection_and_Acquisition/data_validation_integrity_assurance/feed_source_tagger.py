# feed_source_tagger.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
import hashlib
import uuid

class FeedSourceTagger:
    """
    A class to attach metadata to each incoming data row or candle.
    Enables data provenance tracking and source identification.
    """
    
    def __init__(self):
        """Initialize Feed Source Tagger"""
        self.logger = self._setup_logger()
        
        # Feed sources
        self.feed_sources = {}
        self.source_tags = {}
        
        # Tagging status
        self.is_tagging = False
        self.tagging_thread = None
        
        # Tag generation
        self.tag_prefix = "dt_"
        self.include_timestamp = True
        self.include_source_id = True
        self.include_hash = True
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("FeedSourceTagger")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("feed_source_tagger.log")
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
    
    def add_feed_source(self, source_id, source_name, source_type, metadata=None):
        """
        Add feed source
        
        Args:
            source_id (str): Source identifier
            source_name (str): Source display name
            source_type (str): Source type (e.g., 'mt5', 'exchange_api', 'file')
            metadata (dict): Additional source metadata
            
        Returns:
            bool: Whether successfully added source
        """
        if source_id in self.feed_sources:
            self.logger.warning(f"Feed source {source_id} already exists")
            return False
        
        # Add source
        self.feed_sources[source_id] = {
            'source_name': source_name,
            'source_type': source_type,
            'metadata': metadata or {},
            'created_at': datetime.now()
        }
        
        # Initialize source tags
        self.source_tags[source_id] = []
        
        self.logger.info(f"Added feed source {source_id} ({source_name})")
        return True
    
    def remove_feed_source(self, source_id):
        """
        Remove feed source
        
        Args:
            source_id (str): Source identifier
            
        Returns:
            bool: Whether successfully removed source
        """
        if source_id not in self.feed_sources:
            self.logger.warning(f"Feed source {source_id} not found")
            return False
        
        # Remove source and tags
        del self.feed_sources[source_id]
        del self.source_tags[source_id]
        
        self.logger.info(f"Removed feed source {source_id}")
        return True
    
    def set_tag_prefix(self, prefix):
        """
        Set tag prefix
        
        Args:
            prefix (str): Tag prefix
            
        Returns:
            bool: Whether successfully set prefix
        """
        if not isinstance(prefix, str):
            self.logger.error("Tag prefix must be a string")
            return False
        
        self.tag_prefix = prefix
        self.logger.info(f"Set tag prefix to {prefix}")
        return True
    
    def set_include_timestamp(self, include):
        """
        Set whether to include timestamp in tags
        
        Args:
            include (bool): Whether to include timestamp
            
        Returns:
            bool: Whether successfully set option
        """
        if not isinstance(include, bool):
            self.logger.error("Include timestamp must be a boolean")
            return False
        
        self.include_timestamp = include
        self.logger.info(f"Set include timestamp to {include}")
        return True
    
    def set_include_source_id(self, include):
        """
        Set whether to include source ID in tags
        
        Args:
            include (bool): Whether to include source ID
            
        Returns:
            bool: Whether successfully set option
        """
        if not isinstance(include, bool):
            self.logger.error("Include source ID must be a boolean")
            return False
        
        self.include_source_id = include
        self.logger.info(f"Set include source ID to {include}")
        return True
    
    def set_include_hash(self, include):
        """
        Set whether to include hash in tags
        
        Args:
            include (bool): Whether to include hash
            
        Returns:
            bool: Whether successfully set option
        """
        if not isinstance(include, bool):
            self.logger.error("Include hash must be a boolean")
            return False
        
        self.include_hash = include
        self.logger.info(f"Set include hash to {include}")
        return True
    
    def generate_tag(self, source_id, data=None):
        """
        Generate tag for data
        
        Args:
            source_id (str): Source identifier
            data (dict): Data to tag
            
        Returns:
            str: Generated tag or None if source not found
        """
        if source_id not in self.feed_sources:
            self.logger.warning(f"Feed source {source_id} not found")
            return None
        
        # Initialize tag components
        tag_components = [self.tag_prefix]
        
        # Add timestamp if requested
        if self.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            tag_components.append(timestamp)
        
        # Add source ID if requested
        if self.include_source_id:
            tag_components.append(source_id)
        
        # Add UUID for uniqueness
        tag_components.append(str(uuid.uuid4())[:8])
        
        # Add hash if requested and data is provided
        if self.include_hash and data is not None:
            # Convert data to string for hashing
            data_str = str(data)
            
            # Generate hash
            data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
            tag_components.append(data_hash)
        
        # Join components to create tag
        tag = "_".join(tag_components)
        
        # Add to source tags
        self.source_tags[source_id].append({
            'tag': tag,
            'timestamp': datetime.now(),
            'data_hash': data_hash if self.include_hash and data is not None else None
        })
        
        # Limit tag history
        if len(self.source_tags[source_id]) > 1000:
            self.source_tags[source_id] = self.source_tags[source_id][-1000:]
        
        return tag
    
    def tag_data(self, source_id, data):
        """
        Tag data with source information
        
        Args:
            source_id (str): Source identifier
            data (dict): Data to tag
            
        Returns:
            dict: Tagged data or None if source not found
        """
        if source_id not in self.feed_sources:
            self.logger.warning(f"Feed source {source_id} not found")
            return None
        
        # Generate tag
        tag = self.generate_tag(source_id, data)
        
        # Create tagged data
        tagged_data = data.copy()
        
        # Add source metadata
        tagged_data['_source'] = {
            'id': source_id,
            'name': self.feed_sources[source_id]['source_name'],
            'type': self.feed_sources[source_id]['source_type'],
            'metadata': self.feed_sources[source_id]['metadata'].copy(),
            'tag': tag,
            'tagged_at': datetime.now()
        }
        
        return tagged_data
    
    def start_tagging(self):
        """
        Start tagging
        
        Returns:
            bool: Whether successfully started tagging
        """
        if self.is_tagging:
            self.logger.warning("Tagging is already running")
            return False
        
        if not self.feed_sources:
            self.logger.error("No feed sources to tag")
            return False
        
        # Start tagging thread
        self.is_tagging = True
        self.tagging_thread = threading.Thread(target=self._tag_data)
        self.tagging_thread.daemon = True
        self.tagging_thread.start()
        
        self.logger.info("Started feed source tagging")
        return True
    
    def stop_tagging(self):
        """
        Stop tagging
        
        Returns:
            bool: Whether successfully stopped tagging
        """
        if not self.is_tagging:
            self.logger.warning("Tagging is not running")
            return False
        
        # Stop tagging thread
        self.is_tagging = False
        if self.tagging_thread and self.tagging_thread.is_alive():
            self.tagging_thread.join(timeout=5)
        
        self.logger.info("Stopped feed source tagging")
        return True
    
    def _tag_data(self):
        """
        Internal method to tag data, runs in separate thread
        """
        while self.is_tagging:
            try:
                # This is a placeholder for actual tagging logic
                # In a real implementation, this would get data from a queue or callback
                # and tag it with source information
                
                # Sleep for a while
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error tagging data: {str(e)}")
                # Brief sleep before continuing after error
                time.sleep(1)
    
    def get_feed_source(self, source_id):
        """
        Get feed source information
        
        Args:
            source_id (str): Source identifier
            
        Returns:
            dict: Feed source information or None if source not found
        """
        if source_id not in self.feed_sources:
            self.logger.warning(f"Feed source {source_id} not found")
            return None
        
        return self.feed_sources[source_id].copy()
    
    def get_source_tags(self, source_id, count=None):
        """
        Get tags for a source
        
        Args:
            source_id (str): Source identifier
            count (int): Number of tags to get, None means get all
            
        Returns:
            list: List of tags or None if source not found
        """
        if source_id not in self.source_tags:
            self.logger.warning(f"Source tags for {source_id} not found")
            return None
        
        if count is None:
            return self.source_tags[source_id].copy()
        else:
            return self.source_tags[source_id][-count:]
    
    def get_source_tags_dataframe(self, source_id, count=None):
        """
        Get tags for a source as DataFrame
        
        Args:
            source_id (str): Source identifier
            count (int): Number of tags to get, None means get all
            
        Returns:
            pandas.DataFrame: DataFrame containing tags or None if source not found
        """
        tags = self.get_source_tags(source_id, count)
        
        if tags is None:
            return None
        
        if not tags:
            return pd.DataFrame()
        
        df = pd.DataFrame(tags)
        
        # Convert timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def save_tags_to_csv(self, source_id, filename, count=None):
        """
        Save tags for a source to CSV file
        
        Args:
            source_id (str): Source identifier
            filename (str): Filename
            count (int): Number of tags to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_source_tags_dataframe(source_id, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No tags to save for source {source_id}")
                return False
            
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} tags for {source_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save tags for {source_id} to CSV: {str(e)}")
            return False
    
    def save_tags_to_parquet(self, source_id, filename, count=None):
        """
        Save tags for a source to Parquet file
        
        Args:
            source_id (str): Source identifier
            filename (str): Filename
            count (int): Number of tags to save, None means save all
            
        Returns:
            bool: Whether successfully saved
        """
        try:
            df = self.get_source_tags_dataframe(source_id, count)
            
            if df is None or df.empty:
                self.logger.warning(f"No tags to save for source {source_id}")
                return False
            
            df.to_parquet(filename, index=False)
            self.logger.info(f"Saved {len(df)} tags for {source_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save tags for {source_id} to Parquet: {str(e)}")
            return False
    
    def clear_tags(self, source_id):
        """
        Clear tags for a source
        
        Args:
            source_id (str): Source identifier
            
        Returns:
            bool: Whether successfully cleared tags
        """
        if source_id not in self.source_tags:
            self.logger.warning(f"Source tags for {source_id} not found")
            return False
        
        # Clear tags
        self.source_tags[source_id] = []
        
        self.logger.info(f"Cleared tags for source {source_id}")
        return True
