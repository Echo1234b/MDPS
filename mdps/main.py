"""
mdps.main

Unified entrypoint that runs the MDPS pipeline using the dynamically
aggregated components from the mdps package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from . import (
    DataCollector,
    DataCleaner,
    FeatureEngine,
    ChartAnalyzer,
    MarketAnalyzer,
    ExternalFactors,
    PredictionEngine,
    StrategyManager,
)


class MDPSConfig:
    """Lightweight config adapter using the existing root config if available."""

    def __init__(self) -> None:
        try:
            # Reuse legacy configuration if present
            from config import MDPSConfig as RootConfig  # type: ignore

            rc = RootConfig()
            self.__dict__.update(rc.__dict__)
        except Exception:
            self.project_root = Path(__file__).resolve().parents[1]
            self.data_dir = self.project_root / "data"
            self.models_dir = self.project_root / "models"
            self.cleaning_settings = {"smooth_window": 5}


class MDPS:
    def __init__(self) -> None:
        self.config = MDPSConfig()

        self.data_collector = DataCollector(self.config)
        self.data_cleaner = DataCleaner(self.config)
        self.feature_engine = FeatureEngine(self.config)
        self.chart_analyzer = ChartAnalyzer(self.config)
        self.market_analyzer = MarketAnalyzer(self.config)
        self.external_factors = ExternalFactors(self.config)
        self.prediction_engine = PredictionEngine(self.config)
        self.strategy_manager = StrategyManager(self.config)

    def initialize(self) -> None:
        logging.info("MDPS: Initializing components")
        self.data_collector.initialize_feeds()
        self.market_analyzer.initialize()
        self.external_factors.initialize()
        self.prediction_engine.load_models()
        self.strategy_manager.initialize()

    def process_once(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        logging.info(f"MDPS: Processing {symbols} {timeframe}")
        raw = self.data_collector.collect_data(symbols, timeframe)
        clean = self.data_cleaner.process(raw)
        feats = self.feature_engine.generate_features(clean)
        chart = self.chart_analyzer.analyze(clean)
        mkt = self.market_analyzer.analyze_structure(clean)
        ext = self.external_factors.get_current_factors()
        pred = self.prediction_engine.predict(feats, chart, mkt, ext)
        sig = self.strategy_manager.execute_decisions(pred, mkt, ext)
        return {"signals": sig, "predictions": pred, "market_context": mkt, "chart": chart}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("mdps.log")],
    )
    mdps = MDPS()
    mdps.initialize()
    result = mdps.process_once(["EURUSD"], "M5")
    logging.info(
        f"MDPS: Completed one cycle. Signal={result['signals'].get('signal', 'n/a')}"
    )


if __name__ == "__main__":
    main()

