"""
News Sentiment Analyzer

NLP-based sentiment analysis with advanced models, real-time news processing,
sentiment scoring algorithms, and market impact correlation analysis.
"""

import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import hashlib
from collections import deque
import threading
import time

# NLP and ML imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers or NLTK not available. Some features will be limited.")

class SentimentScore(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

@dataclass
class NewsArticle:
    """News article data structure"""
    id: str
    title: str
    content: str
    source: str
    author: Optional[str]
    published_at: datetime
    url: Optional[str]
    category: Optional[str]
    symbols_mentioned: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[SentimentScore] = None
    confidence: Optional[float] = None
    keywords: List[str] = field(default_factory=list)
    processed_at: Optional[datetime] = None

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    article_id: str
    symbol: Optional[str]
    overall_sentiment: float  # -1 to 1
    sentiment_label: SentimentScore
    confidence: float  # 0 to 1
    title_sentiment: float
    content_sentiment: float
    keyword_sentiment: Dict[str, float]
    model_scores: Dict[str, float]  # Scores from different models
    processing_time: float
    timestamp: datetime

@dataclass
class MarketImpactCorrelation:
    """Market impact correlation analysis"""
    symbol: str
    sentiment_score: float
    price_change_1h: Optional[float]
    price_change_4h: Optional[float]
    price_change_24h: Optional[float]
    volume_change_1h: Optional[float]
    volume_change_4h: Optional[float]
    volume_change_24h: Optional[float]
    correlation_1h: Optional[float]
    correlation_4h: Optional[float]
    correlation_24h: Optional[float]
    sample_size: int
    timestamp: datetime

class NewsSentimentAnalyzer:
    """
    Advanced news sentiment analyzer with multiple NLP models and market correlation analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize News Sentiment Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Models and analyzers
        self.vader_analyzer = None
        self.transformer_pipeline = None
        self.tokenizer = None
        self.lemmatizer = None
        self.stop_words = set()
        
        # Data storage
        self.articles_buffer = deque(maxlen=10000)
        self.sentiment_history = deque(maxlen=50000)
        self.correlation_data = {}
        
        # Callbacks
        self.sentiment_callbacks: List[Callable] = []
        self.correlation_callbacks: List[Callable] = []
        
        # Threading
        self.processing_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'articles_processed': 0,
            'processing_time_avg': 0.0,
            'model_accuracy': {},
            'error_count': 0
        }
        
        # Market data integration
        self.market_data_callback = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models and tools"""
        try:
            # Initialize VADER
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            if TRANSFORMERS_AVAILABLE:
                # Download required NLTK data
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('wordnet', quiet=True)
                    nltk.download('vader_lexicon', quiet=True)
                except:
                    pass
                
                # Initialize NLTK tools
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                
                # Initialize transformer model
                model_name = self.config.get('transformer_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
                try:
                    self.transformer_pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        tokenizer=model_name,
                        return_all_scores=True
                    )
                    self.logger.info(f"Loaded transformer model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load transformer model: {str(e)}")
                    # Fallback to simpler model
                    try:
                        self.transformer_pipeline = pipeline("sentiment-analysis", return_all_scores=True)
                        self.logger.info("Loaded default sentiment analysis model")
                    except Exception as e2:
                        self.logger.error(f"Failed to load any transformer model: {str(e2)}")
            
            self.logger.info("Sentiment analysis models initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords
            
        Returns:
            List[str]: Extracted keywords
        """
        if not text or not TRANSFORMERS_AVAILABLE:
            return []
        
        try:
            # Tokenize and remove stop words
            tokens = word_tokenize(text.lower())
            keywords = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token.isalpha() and token not in self.stop_words and len(token) > 2
            ]
            
            # Count frequency and return most common
            from collections import Counter
            keyword_counts = Counter(keywords)
            return [word for word, count in keyword_counts.most_common(max_keywords)]
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_symbols(self, text: str) -> List[str]:
        """
        Extract cryptocurrency/stock symbols from text
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Extracted symbols
        """
        symbols = []
        
        # Common crypto symbols pattern
        crypto_pattern = r'\b(?:BTC|ETH|ADA|DOT|LINK|UNI|AAVE|SOL|AVAX|MATIC|FTM|NEAR|ATOM|ALGO|XRP|LTC|BCH|ETC|XLM|TRX|VET|THETA|FIL|EOS|XTZ|DASH|ZEC|QTUM|NEO|ONT|ICX|ZIL|BAT|ENJ|MANA|SAND|AXS|CRV|YFI|COMP|MKR|SNX|SUSHI|1INCH)\b'
        
        # Stock symbols pattern (3-5 uppercase letters)
        stock_pattern = r'\b[A-Z]{3,5}\b'
        
        # Crypto symbols with $ prefix
        crypto_dollar_pattern = r'\$([A-Z]{2,10})\b'
        
        # Extract patterns
        symbols.extend(re.findall(crypto_pattern, text.upper()))
        symbols.extend(re.findall(stock_pattern, text))
        symbols.extend(re.findall(crypto_dollar_pattern, text.upper()))
        
        # Remove duplicates and common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'HAS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        symbols = list(set([s for s in symbols if s not in false_positives]))
        
        return symbols
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
            
        Returns:
            Dict: VADER sentiment scores
        """
        if not self.vader_analyzer or not text:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            self.logger.error(f"VADER analysis error: {str(e)}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
    
    def analyze_sentiment_transformer(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using transformer model
        
        Args:
            text: Input text
            
        Returns:
            Dict: Transformer sentiment scores
        """
        if not self.transformer_pipeline or not text:
            return {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            results = self.transformer_pipeline(text)
            
            # Convert to standardized format
            scores = {'negative': 0.0, 'neutral': 0.0, 'positive': 0.0}
            
            if isinstance(results, list) and len(results) > 0:
                for result in results[0]:  # First result contains all scores
                    label = result['label'].lower()
                    score = result['score']
                    
                    if 'neg' in label:
                        scores['negative'] = score
                    elif 'pos' in label:
                        scores['positive'] = score
                    else:
                        scores['neutral'] = score
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Transformer analysis error: {str(e)}")
            return {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
    
    def calculate_overall_sentiment(self, title: str, content: str) -> Tuple[float, float, SentimentScore]:
        """
        Calculate overall sentiment score combining multiple models
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            Tuple: (overall_sentiment, confidence, sentiment_label)
        """
        try:
            # Preprocess texts
            title_clean = self.preprocess_text(title)
            content_clean = self.preprocess_text(content)
            combined_text = f"{title_clean} {content_clean}"
            
            # VADER analysis
            title_vader = self.analyze_sentiment_vader(title_clean)
            content_vader = self.analyze_sentiment_vader(content_clean)
            
            # Transformer analysis
            title_transformer = self.analyze_sentiment_transformer(title_clean)
            content_transformer = self.analyze_sentiment_transformer(content_clean)
            
            # Calculate weighted sentiment scores
            title_weight = 0.3
            content_weight = 0.7
            
            # VADER compound score (-1 to 1)
            vader_sentiment = (title_vader['compound'] * title_weight + 
                             content_vader['compound'] * content_weight)
            
            # Transformer score (convert to -1 to 1 scale)
            title_trans_sentiment = (title_transformer['positive'] - title_transformer['negative'])
            content_trans_sentiment = (content_transformer['positive'] - content_transformer['negative'])
            transformer_sentiment = (title_trans_sentiment * title_weight + 
                                   content_trans_sentiment * content_weight)
            
            # Combine models with equal weight
            overall_sentiment = (vader_sentiment + transformer_sentiment) / 2
            
            # Calculate confidence based on model agreement
            agreement = 1 - abs(vader_sentiment - transformer_sentiment) / 2
            base_confidence = min(
                title_vader.get('pos', 0) + title_vader.get('neg', 0),
                title_transformer.get('positive', 0) + title_transformer.get('negative', 0)
            )
            confidence = (agreement + base_confidence) / 2
            
            # Determine sentiment label
            if overall_sentiment >= 0.5:
                sentiment_label = SentimentScore.VERY_POSITIVE
            elif overall_sentiment >= 0.1:
                sentiment_label = SentimentScore.POSITIVE
            elif overall_sentiment <= -0.5:
                sentiment_label = SentimentScore.VERY_NEGATIVE
            elif overall_sentiment <= -0.1:
                sentiment_label = SentimentScore.NEGATIVE
            else:
                sentiment_label = SentimentScore.NEUTRAL
            
            return overall_sentiment, confidence, sentiment_label
            
        except Exception as e:
            self.logger.error(f"Error calculating overall sentiment: {str(e)}")
            return 0.0, 0.0, SentimentScore.NEUTRAL
    
    def analyze_article(self, article: NewsArticle) -> SentimentAnalysis:
        """
        Analyze sentiment of a news article
        
        Args:
            article: News article to analyze
            
        Returns:
            SentimentAnalysis: Sentiment analysis result
        """
        start_time = time.time()
        
        try:
            # Calculate overall sentiment
            overall_sentiment, confidence, sentiment_label = self.calculate_overall_sentiment(
                article.title, article.content
            )
            
            # Analyze title and content separately
            title_vader = self.analyze_sentiment_vader(self.preprocess_text(article.title))
            content_vader = self.analyze_sentiment_vader(self.preprocess_text(article.content))
            title_transformer = self.analyze_sentiment_transformer(self.preprocess_text(article.title))
            content_transformer = self.analyze_sentiment_transformer(self.preprocess_text(article.content))
            
            # Extract keywords and analyze their sentiment
            keywords = self.extract_keywords(f"{article.title} {article.content}")
            keyword_sentiment = {}
            for keyword in keywords:
                kw_scores = self.analyze_sentiment_vader(keyword)
                keyword_sentiment[keyword] = kw_scores['compound']
            
            # Create analysis result
            analysis = SentimentAnalysis(
                article_id=article.id,
                symbol=article.symbols_mentioned[0] if article.symbols_mentioned else None,
                overall_sentiment=overall_sentiment,
                sentiment_label=sentiment_label,
                confidence=confidence,
                title_sentiment=title_vader['compound'],
                content_sentiment=content_vader['compound'],
                keyword_sentiment=keyword_sentiment,
                model_scores={
                    'vader_title': title_vader['compound'],
                    'vader_content': content_vader['compound'],
                    'transformer_title': title_transformer['positive'] - title_transformer['negative'],
                    'transformer_content': content_transformer['positive'] - content_transformer['negative']
                },
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
            # Update article with sentiment info
            article.sentiment_score = overall_sentiment
            article.sentiment_label = sentiment_label
            article.confidence = confidence
            article.keywords = keywords
            article.processed_at = datetime.now()
            
            # Store results
            with self.lock:
                self.sentiment_history.append(analysis)
                self.metrics['articles_processed'] += 1
                self.metrics['processing_time_avg'] = (
                    (self.metrics['processing_time_avg'] * (self.metrics['articles_processed'] - 1) + 
                     analysis.processing_time) / self.metrics['articles_processed']
                )
            
            # Trigger callbacks
            self._trigger_sentiment_callbacks(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing article {article.id}: {str(e)}")
            self.metrics['error_count'] += 1
            
            # Return neutral analysis
            return SentimentAnalysis(
                article_id=article.id,
                symbol=None,
                overall_sentiment=0.0,
                sentiment_label=SentimentScore.NEUTRAL,
                confidence=0.0,
                title_sentiment=0.0,
                content_sentiment=0.0,
                keyword_sentiment={},
                model_scores={},
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def add_article(self, article: NewsArticle) -> SentimentAnalysis:
        """
        Add article for processing
        
        Args:
            article: News article to process
            
        Returns:
            SentimentAnalysis: Sentiment analysis result
        """
        # Extract symbols if not already done
        if not article.symbols_mentioned:
            article.symbols_mentioned = self.extract_symbols(f"{article.title} {article.content}")
        
        # Analyze sentiment
        analysis = self.analyze_article(article)
        
        # Store article
        with self.lock:
            self.articles_buffer.append(article)
        
        return analysis
    
    def get_sentiment_history(self, symbol: Optional[str] = None, 
                            hours: int = 24) -> List[SentimentAnalysis]:
        """
        Get sentiment analysis history
        
        Args:
            symbol: Filter by symbol (optional)
            hours: Number of hours to look back
            
        Returns:
            List[SentimentAnalysis]: Historical sentiment data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            history = [
                analysis for analysis in self.sentiment_history
                if analysis.timestamp >= cutoff_time and 
                (symbol is None or analysis.symbol == symbol)
            ]
        
        return sorted(history, key=lambda x: x.timestamp)
    
    def calculate_sentiment_trend(self, symbol: Optional[str] = None, 
                                hours: int = 24) -> Dict[str, float]:
        """
        Calculate sentiment trend over time
        
        Args:
            symbol: Filter by symbol (optional)
            hours: Number of hours to analyze
            
        Returns:
            Dict: Trend analysis results
        """
        history = self.get_sentiment_history(symbol, hours)
        
        if len(history) < 2:
            return {
                'current_sentiment': 0.0,
                'trend_slope': 0.0,
                'volatility': 0.0,
                'sample_size': len(history)
            }
        
        # Calculate trend
        sentiments = [h.overall_sentiment for h in history]
        timestamps = [(h.timestamp - history[0].timestamp).total_seconds() / 3600 for h in history]
        
        # Linear regression for trend
        if len(sentiments) > 1:
            trend_slope = np.polyfit(timestamps, sentiments, 1)[0]
        else:
            trend_slope = 0.0
        
        return {
            'current_sentiment': sentiments[-1],
            'average_sentiment': np.mean(sentiments),
            'trend_slope': trend_slope,
            'volatility': np.std(sentiments),
            'min_sentiment': np.min(sentiments),
            'max_sentiment': np.max(sentiments),
            'sample_size': len(history)
        }
    
    def add_sentiment_callback(self, callback: Callable[[SentimentAnalysis], None]):
        """Add callback for sentiment analysis results"""
        self.sentiment_callbacks.append(callback)
    
    def add_correlation_callback(self, callback: Callable[[MarketImpactCorrelation], None]):
        """Add callback for correlation analysis results"""
        self.correlation_callbacks.append(callback)
    
    def set_market_data_callback(self, callback: Callable[[str], Dict[str, float]]):
        """Set callback to get market data for correlation analysis"""
        self.market_data_callback = callback
    
    def _trigger_sentiment_callbacks(self, analysis: SentimentAnalysis):
        """Trigger sentiment analysis callbacks"""
        for callback in self.sentiment_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                self.logger.error(f"Sentiment callback error: {str(e)}")
    
    def _trigger_correlation_callbacks(self, correlation: MarketImpactCorrelation):
        """Trigger correlation analysis callbacks"""
        for callback in self.correlation_callbacks:
            try:
                callback(correlation)
            except Exception as e:
                self.logger.error(f"Correlation callback error: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of loaded models"""
        return {
            'vader_available': self.vader_analyzer is not None,
            'transformer_available': self.transformer_pipeline is not None,
            'nltk_available': TRANSFORMERS_AVAILABLE and self.lemmatizer is not None,
            'transformers_library': TRANSFORMERS_AVAILABLE
        }