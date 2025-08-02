# config.py

class Config:
    class QA:
        class MissingHandler:
            strategy = 'ffill'
        class OutlierDetector:
            method = 'z_score'
            threshold = 3.0
            handling_method = 'cap'
        
        missing_handler = MissingHandler()
        outlier_detector = OutlierDetector()
        
    class TSA:
        class TimestampNormalizer:
            timestamp_col = 'time'
            timezone = 'UTC'
        class FreqConverter:
            freq = '4H'
        
        timestamp_normalizer = TimestampNormalizer()
        freq_converter = FreqConverter()
        
    class NST:
        class NoiseFilter:
            window_length = 5
            polyorder = 2
        class DataSmoother:
            ema_span = 14
        class VolumeNormalizer:
            method = 'robust' # 'robust' أو 'standard'
        
        noise_filter = NoiseFilter()
        data_smoother = DataSmoother()
        volume_normalizer = VolumeNormalizer()
        
    class CSA:
        class PriceAnnotator:
            breakout_window = 20
        class MarketClassifier:
            n_clusters = 4
        class ContextEnricher:
            volatility_window = 20
        class AnomalyDetector:
            contamination = 'auto'

        price_annotator = PriceAnnotator()
        market_classifier = MarketClassifier()
        context_enricher = ContextEnricher()
        anomaly_detector = AnomalyDetector()

    qa = QA()
    tsa = TSA()
    nst = NST()
    csa = CSA()

# إنشاء كائن الإعدادات لاستخدامه
config = Config()