"""
mdps package

Aggregates core MDPS components from the existing repository structure and
exposes a clean, importable API:

 - DataCollector
 - DataCleaner
 - FeatureEngine
 - ChartAnalyzer
 - MarketAnalyzer
 - ExternalFactors
 - PredictionEngine
 - StrategyManager

This module dynamically loads implementations that live in legacy folders
whose names may not be valid Python identifiers (contain spaces or symbols).
It guarantees the same import surface regardless of layout.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_module_from_path(module_name: str, path: Path):
    """Safely load a module from an arbitrary file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module
        raise ImportError(f"Could not create spec for {module_name} at {path}")
    except Exception as exc:
        logging.error(f"Failed to load module {module_name} from {path}: {exc}")
        raise


# DataCollector
try:
    # Normal import from sanitized legacy package
    from Data_Collection_and_Acquisition import DataCollector  # type: ignore
except Exception:
    # As a fallback, expose a minimal placeholder
    logging.warning("DataCollector not found; exposing a minimal placeholder.")

    class DataCollector:  # type: ignore
        def __init__(self, config=None):
            self.config = config

        def initialize_feeds(self):
            logging.info("DataCollector placeholder - initialize_feeds")
            return True

        def collect_data(self, symbols, timeframe):
            logging.info(
                f"DataCollector placeholder - collect_data for {symbols} {timeframe}"
            )
            # Minimal structure to keep the pipeline running without pandas
            from datetime import datetime, timedelta
            data = []
            base = datetime.utcnow()
            for i in range(50):
                ts = base + timedelta(minutes=i * 5)
                data.append(
                    {
                        "timestamp": ts,
                        "open": 1.0,
                        "high": 1.01,
                        "low": 0.99,
                        "close": 1.0,
                        "volume": 1000,
                    }
                )
            return data


# DataCleaner (legacy folder has spaces in name)
try:
    _dsp = _load_module_from_path(
        "mdps.data_cleaning_signal_processing",
        ROOT_DIR / "Data Cleaning & Signal Processing" / "__init__.py",
    )
    DataCleaner = getattr(_dsp, "DataCleaner")  # type: ignore
except Exception:
    logging.warning("DataCleaner not found; exposing a minimal placeholder.")

    class DataCleaner:  # type: ignore
        def __init__(self, config=None):
            self.config = config

        def process(self, data):
            return data


# FeatureEngine (legacy folder has spaces in name)
try:
    _pfe = _load_module_from_path(
        "mdps.preprocessing_feature_engineering",
        ROOT_DIR / "Preprocessing & Feature Engineering" / "__init__.py",
    )
    FeatureEngine = getattr(_pfe, "FeatureEngine")  # type: ignore
except Exception:
    logging.warning("FeatureEngine not found; exposing a minimal placeholder.")

    class FeatureEngine:  # type: ignore
        def __init__(self, config=None):
            self.config = config

        def generate_features(self, data):
            return data


# ChartAnalyzer (legacy folder has spaces in name)
try:
    _cat = _load_module_from_path(
        "mdps.advanced_chart_analysis_tools",
        ROOT_DIR / "Advanced Chart Analysis Tools" / "__init__.py",
    )
    ChartAnalyzer = getattr(_cat, "ChartAnalyzer")  # type: ignore
except Exception:
    logging.warning("ChartAnalyzer not found; exposing a minimal placeholder.")

    class ChartAnalyzer:  # type: ignore
        def __init__(self, config=None):
            self.config = config

        def analyze(self, data):
            return {"patterns": [], "signals": []}


# MarketAnalyzer (no clean aggregator exists; provide a light wrapper)
class MarketAnalyzer:  # type: ignore
    def __init__(self, config=None):
        self.config = config

    def initialize(self):
        logging.info("MarketAnalyzer initialized")
        return True

    def analyze_structure(self, data):
        # Very light heuristic to keep the pipeline working
        trend = "sideways"
        volatility = "normal"
        try:
            import pandas as pd  # noqa: F401

            if hasattr(data, "__len__") and len(data) >= 2:
                # If close increased, mark uptrend, else downtrend
                last_close = data["close"].iloc[-1]
                first_close = data["close"].iloc[0]
                if last_close > first_close * 1.002:
                    trend = "uptrend"
                elif last_close < first_close * 0.998:
                    trend = "downtrend"
        except Exception:
            pass

        return {"trend": trend, "volatility": volatility, "regime": "ranging"}


# ExternalFactors (light wrapper; detailed modules are available under legacy folder)
class ExternalFactors:  # type: ignore
    def __init__(self, config=None):
        self.config = config

    def initialize(self):
        logging.info("ExternalFactors initialized")
        return True

    def get_current_factors(self):
        return {
            "sentiment": 0.5,
            "news_impact": "medium",
            "economic_data": {"gdp_growth": 0.0},
        }


# PredictionEngine (legacy folder has symbols in name)
try:
    _pe = _load_module_from_path(
        "mdps.prediction_engine",
        ROOT_DIR / "Prediction Engine (MLDL Models)" / "__init__.py",
    )
    PredictionEngine = getattr(_pe, "PredictionEngine")  # type: ignore
except Exception:
    logging.warning("PredictionEngine not found; exposing a minimal placeholder.")

    class PredictionEngine:  # type: ignore
        def __init__(self, config=None):
            self.config = config
            self.models_loaded = True

        def load_models(self):
            return True

        def predict(self, features, chart_patterns, market_context, external_data):
            return {"direction": "hold", "confidence": 0.5}


# StrategyManager from Strategy & Decision Layer (legacy folder has spaces)
try:
    _sdl = _load_module_from_path(
        "mdps.strategy_decision_layer_main",
        ROOT_DIR / "Strategy & Decision Layer" / "main.py",
    )

    class StrategyManager:  # type: ignore
        def __init__(self, config=None):
            self._impl = getattr(_sdl, "StrategyDecisionLayer")(config)

        def initialize(self):
            logging.info("StrategyManager initialized")
            return True

        def execute_decisions(self, predictions, market_context, external_data):
            # Map inputs to the StrategyDecisionLayer pipeline expectations
            signal = {
                "predictions": predictions,
                "market_context": market_context,
                "external_data": external_data,
            }
            trade = self._impl.process_signal(signal)
            if trade:
                return {"signal": "buy", "strength": predictions.get("confidence", 0.5)}
            return {"signal": "hold", "strength": 0.0}

except Exception:
    logging.warning("StrategyManager not found; exposing a minimal placeholder.")

    class StrategyManager:  # type: ignore
        def __init__(self, config=None):
            self.config = config

        def initialize(self):
            return True

        def execute_decisions(self, predictions, market_context, external_data):
            return {"signal": predictions.get("direction", "hold"), "strength": predictions.get("confidence", 0.5)}


__all__ = [
    "DataCollector",
    "DataCleaner",
    "FeatureEngine",
    "ChartAnalyzer",
    "MarketAnalyzer",
    "ExternalFactors",
    "PredictionEngine",
    "StrategyManager",
]

