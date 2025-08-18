# data_processing/main.py
"""
Main Pipeline Module - Fluent API Design

This module provides the main DataCleaningPipeline class that integrates all
data cleaning and signal processing components using a flexible, chainable interface.
"""
import logging
import pandas as pd
from .data_quality_assurance import MissingValueHandler, DuplicateEntryRemover, OutlierDetector, DataSanitizer
from .temporal_structural_alignment import TimestampNormalizer, DataFrequencyConverter
from .noise_signal_treatment import DataSmoother, VolumeNormalizer
# ... (يمكن إضافة بقية الاستيرادات عند الحاجة)

logger = logging.getLogger(__name__)

class DataCleaningPipeline:
    """
    A configurable pipeline with a Fluent API to clean, process, and enrich financial data.
    Allows for dynamic and readable construction of processing workflows.
    """
    def __init__(self, config):
        logger.info("Initializing DataCleaningPipeline with provided configuration.")
        self.config = config
        self.df = None

        # تهيئة المكونات مرة واحدة
        self.qa = {
            'missing_handler': MissingValueHandler(self.config.qa.missing_handler),
            'duplicate_remover': DuplicateEntryRemover(),
            'outlier_detector': OutlierDetector(self.config.qa.outlier_detector),
            'sanitizer': DataSanitizer()
        }
        self.tsa = {
            'timestamp_normalizer': TimestampNormalizer(self.config.tsa.timestamp_normalizer),
            'freq_converter': DataFrequencyConverter(self.config.tsa.freq_converter)
        }
        self.nst = {
            'data_smoother': DataSmoother(self.config.nst.data_smoother),
            'volume_normalizer': VolumeNormalizer(self.config.nst.volume_normalizer)
        }

    def start_with(self, df: pd.DataFrame):
        """Starts the pipeline with a new DataFrame."""
        self.df = df.copy()
        return self

    def sanitize_data(self):
        """Applies basic data sanitization and type optimization."""
        if self.df is None: raise ValueError("Pipeline not started. Use start_with(df).")
        self.df = self.qa['sanitizer'].sanitize(self.df)
        return self

    def handle_missing_values(self):
        """Applies the configured missing value handler."""
        if self.df is None: raise ValueError("Pipeline not started.")
        self.df = self.qa['missing_handler'].handle(self.df)
        return self
        
    def remove_duplicates(self):
        """Removes duplicate entries."""
        if self.df is None: raise ValueError("Pipeline not started.")
        self.df = self.qa['duplicate_remover'].remove(self.df)
        return self
        
    def normalize_timestamps(self):
        """Normalizes and sets the datetime index."""
        if self.df is None: raise ValueError("Pipeline not started.")
        self.df = self.tsa['timestamp_normalizer'].normalize(self.df)
        return self

    def convert_frequency(self):
        """Resamples the data to the target frequency."""
        if self.df is None: raise ValueError("Pipeline not started.")
        self.df = self.tsa['freq_converter'].convert(self.df)
        return self
        
    def smooth_data(self, columns: list):
        """Applies EMA smoothing to the specified columns."""
        if self.df is None: raise ValueError("Pipeline not started.")
        for col in columns:
            self.df[f'{col}_smoothed'] = self.nst['data_smoother'].smooth(self.df[col])
        return self

    def normalize_volume(self, columns=['volume']):
        """Applies normalization to volume columns."""
        if self.df is None: raise ValueError("Pipeline not started.")
        self.df = self.nst['volume_normalizer'].normalize(self.df, columns)
        return self
        
    def get_dataframe(self) -> pd.DataFrame:
        """Returns the final processed DataFrame."""
        if self.df is None: raise ValueError("Pipeline not started.")
        logger.info(f"Pipeline finished. Returning DataFrame with shape {self.df.shape}")
        return self.df