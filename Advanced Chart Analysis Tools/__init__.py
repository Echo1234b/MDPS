"""
Advanced Chart Analysis Tools Module
Provides comprehensive chart pattern recognition, technical analysis, and signal generation.
"""

import logging
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional

class ChartAnalyzer:
    """Main chart analysis class combining multiple pattern recognition techniques"""
    
    def __init__(self, config=None):
        self.config = config
        self.pattern_recognizer = ChartPatternRecognizer()
        self.support_resistance_finder = SupportResistanceFinder()
        self.elliott_wave_analyzer = ElliottWaveAnalyzer()
        self.fibonacci_toolkit = FibonacciToolkit()
        self.harmonic_pattern_detector = HarmonicPatternDetector()
        self.price_action_annotator = PriceActionAnnotator()
        
    def analyze(self, data):
        """Perform comprehensive chart analysis"""
        logging.info("ChartAnalyzer: Starting comprehensive chart analysis")
        
        try:
            analysis_results = {
                'patterns': [],
                'signals': [],
                'support_resistance': [],
                'fibonacci_levels': [],
                'price_action': []
            }
            
            # 1. Detect chart patterns
            chart_patterns = self.pattern_recognizer.detect_patterns(data)
            analysis_results['patterns'].extend(chart_patterns)
            
            # 2. Find support and resistance levels
            sr_levels = self.support_resistance_finder.find_levels(data)
            analysis_results['support_resistance'] = sr_levels
            
            # 3. Calculate Fibonacci levels
            fib_levels = self.fibonacci_toolkit.calculate_levels(data)
            analysis_results['fibonacci_levels'] = fib_levels
            
            # 4. Detect harmonic patterns
            harmonic_patterns = self.harmonic_pattern_detector.detect_patterns(data)
            analysis_results['patterns'].extend(harmonic_patterns)
            
            # 5. Analyze price action
            price_action = self.price_action_annotator.analyze_action(data)
            analysis_results['price_action'] = price_action
            
            # 6. Generate trading signals based on analysis
            signals = self._generate_signals(analysis_results, data)
            analysis_results['signals'] = signals
            
            logging.info(f"ChartAnalyzer: Found {len(analysis_results['patterns'])} patterns, {len(signals)} signals")
            return analysis_results
            
        except Exception as e:
            logging.error(f"ChartAnalyzer: Error in analysis: {e}")
            return {'patterns': [], 'signals': [], 'support_resistance': [], 'fibonacci_levels': [], 'price_action': []}

    def _generate_signals(self, analysis_results, data):
        """Generate trading signals based on analysis results"""
        signals = []
        
        # Signal from pattern confluence
        pattern_count = len(analysis_results['patterns'])
        if pattern_count >= 2:
            signals.append({
                'type': 'pattern_confluence',
                'strength': min(pattern_count * 0.2, 1.0),
                'description': f'Multiple patterns detected: {pattern_count}'
            })
        
        # Signal from support/resistance test
        if analysis_results['support_resistance']:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 0
            for level in analysis_results['support_resistance']:
                distance = abs(current_price - level['price']) / current_price
                if distance < 0.01:  # Within 1% of level
                    signals.append({
                        'type': 'sr_test',
                        'strength': level['strength'],
                        'description': f'Price testing {level["type"]} at {level["price"]:.4f}'
                    })
        
        return signals

class ChartPatternRecognizer:
    """Recognizes classic chart patterns"""
    
    def detect_patterns(self, data):
        """Detect various chart patterns"""
        logging.info("ChartPatternRecognizer: Detecting chart patterns")
        
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        # Detect double top/bottom patterns
        patterns.extend(self._detect_double_top_bottom(data))
        
        # Detect head and shoulders
        patterns.extend(self._detect_head_shoulders(data))
        
        # Detect triangles
        patterns.extend(self._detect_triangles(data))
        
        # Detect flags and pennants
        patterns.extend(self._detect_flags_pennants(data))
        
        return patterns

    def _detect_double_top_bottom(self, data):
        """Detect double top and double bottom patterns"""
        patterns = []
        
        # Find peaks and troughs
        highs = data['high'].values
        lows = data['low'].values
        
        # Find peaks (potential tops)
        peaks, _ = find_peaks(highs, distance=10, prominence=np.std(highs) * 0.5)
        
        # Find troughs (potential bottoms)
        troughs, _ = find_peaks(-lows, distance=10, prominence=np.std(lows) * 0.5)
        
        # Check for double tops
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1_price = highs[peaks[i]]
                peak2_price = highs[peaks[i + 1]]
                price_diff = abs(peak1_price - peak2_price) / peak1_price
                
                if price_diff < 0.02:  # Within 2% - potential double top
                    patterns.append({
                        'type': 'double_top',
                        'confidence': 0.7,
                        'start_idx': peaks[i],
                        'end_idx': peaks[i + 1],
                        'description': f'Double top pattern detected'
                    })
        
        # Check for double bottoms
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                trough1_price = lows[troughs[i]]
                trough2_price = lows[troughs[i + 1]]
                price_diff = abs(trough1_price - trough2_price) / trough1_price
                
                if price_diff < 0.02:  # Within 2% - potential double bottom
                    patterns.append({
                        'type': 'double_bottom',
                        'confidence': 0.7,
                        'start_idx': troughs[i],
                        'end_idx': troughs[i + 1],
                        'description': f'Double bottom pattern detected'
                    })
        
        return patterns

    def _detect_head_shoulders(self, data):
        """Detect head and shoulders patterns"""
        patterns = []
        
        highs = data['high'].values
        peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.3)
        
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                left_shoulder = highs[peaks[i]]
                head = highs[peaks[i + 1]]
                right_shoulder = highs[peaks[i + 2]]
                
                # Check head and shoulders criteria
                if (head > left_shoulder * 1.02 and head > right_shoulder * 1.02 and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                    
                    patterns.append({
                        'type': 'head_shoulders',
                        'confidence': 0.8,
                        'start_idx': peaks[i],
                        'end_idx': peaks[i + 2],
                        'description': 'Head and shoulders pattern detected'
                    })
        
        return patterns

    def _detect_triangles(self, data):
        """Detect triangle patterns"""
        patterns = []
        
        if len(data) < 30:
            return patterns
        
        # Simple triangle detection based on converging trend lines
        recent_data = data.tail(30)
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Check for ascending triangle (flat top, rising bottom)
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if abs(high_trend) < 0.001 and low_trend > 0.001:
            patterns.append({
                'type': 'ascending_triangle',
                'confidence': 0.6,
                'description': 'Ascending triangle pattern detected'
            })
        elif abs(low_trend) < 0.001 and high_trend < -0.001:
            patterns.append({
                'type': 'descending_triangle',
                'confidence': 0.6,
                'description': 'Descending triangle pattern detected'
            })
        elif high_trend < -0.001 and low_trend > 0.001:
            patterns.append({
                'type': 'symmetrical_triangle',
                'confidence': 0.6,
                'description': 'Symmetrical triangle pattern detected'
            })
        
        return patterns

    def _detect_flags_pennants(self, data):
        """Detect flag and pennant patterns"""
        patterns = []
        
        # Simplified flag detection - look for consolidation after strong move
        if len(data) < 20:
            return patterns
        
        recent_data = data.tail(20)
        price_range = recent_data['close'].max() - recent_data['close'].min()
        avg_price = recent_data['close'].mean()
        
        # Check for tight consolidation (flag characteristic)
        if price_range / avg_price < 0.02:  # Less than 2% range
            # Check for prior strong move
            prior_data = data.tail(40).head(20)
            prior_change = abs(prior_data['close'].iloc[-1] - prior_data['close'].iloc[0]) / prior_data['close'].iloc[0]
            
            if prior_change > 0.05:  # Greater than 5% move
                pattern_type = 'bullish_flag' if prior_data['close'].iloc[-1] > prior_data['close'].iloc[0] else 'bearish_flag'
                patterns.append({
                    'type': pattern_type,
                    'confidence': 0.6,
                    'description': f'{pattern_type.replace("_", " ").title()} pattern detected'
                })
        
        return patterns

class SupportResistanceFinder:
    """Finds dynamic support and resistance levels"""
    
    def find_levels(self, data, lookback=50):
        """Find support and resistance levels"""
        logging.info("SupportResistanceFinder: Finding support and resistance levels")
        
        levels = []
        
        if len(data) < lookback:
            return levels
        
        recent_data = data.tail(lookback)
        
        # Find local maxima (resistance)
        highs = recent_data['high'].values
        resistance_peaks, _ = find_peaks(highs, distance=5)
        
        for peak in resistance_peaks:
            price = highs[peak]
            # Calculate strength based on how many times price approached this level
            strength = self._calculate_level_strength(data, price, 'resistance')
            levels.append({
                'type': 'resistance',
                'price': price,
                'strength': strength,
                'touches': strength
            })
        
        # Find local minima (support)
        lows = recent_data['low'].values
        support_troughs, _ = find_peaks(-lows, distance=5)
        
        for trough in support_troughs:
            price = lows[trough]
            strength = self._calculate_level_strength(data, price, 'support')
            levels.append({
                'type': 'support',
                'price': price,
                'strength': strength,
                'touches': strength
            })
        
        # Sort by strength
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return levels[:10]  # Return top 10 levels

    def _calculate_level_strength(self, data, price, level_type):
        """Calculate the strength of a support/resistance level"""
        tolerance = 0.005  # 0.5% tolerance
        touches = 0
        
        if level_type == 'resistance':
            for high in data['high']:
                if abs(high - price) / price <= tolerance:
                    touches += 1
        else:  # support
            for low in data['low']:
                if abs(low - price) / price <= tolerance:
                    touches += 1
        
        return min(touches / 10.0, 1.0)  # Normalize to 0-1 scale

class ElliottWaveAnalyzer:
    """Basic Elliott Wave analysis"""
    
    def analyze_waves(self, data):
        """Analyze Elliott Wave patterns"""
        logging.info("ElliottWaveAnalyzer: Analyzing Elliott Waves (placeholder)")
        # Placeholder implementation
        return []

class FibonacciToolkit:
    """Fibonacci analysis tools"""
    
    def calculate_levels(self, data):
        """Calculate Fibonacci retracement levels"""
        logging.info("FibonacciToolkit: Calculating Fibonacci levels")
        
        if len(data) < 20:
            return []
        
        # Find recent high and low
        recent_data = data.tail(50)
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        
        # Calculate Fibonacci levels
        diff = high - low
        levels = []
        
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            level = high - (diff * ratio)
            levels.append({
                'level': level,
                'ratio': ratio,
                'type': 'retracement'
            })
        
        return levels

class HarmonicPatternDetector:
    """Detects harmonic patterns like Gartley, Bat, etc."""
    
    def detect_patterns(self, data):
        """Detect harmonic patterns"""
        logging.info("HarmonicPatternDetector: Detecting harmonic patterns (placeholder)")
        # Placeholder implementation
        return []

class PriceActionAnnotator:
    """Analyzes price action for key formations"""
    
    def analyze_action(self, data):
        """Analyze price action patterns"""
        logging.info("PriceActionAnnotator: Analyzing price action")
        
        if len(data) < 10:
            return []
        
        actions = []
        
        # Detect pin bars
        for i in range(1, len(data)):
            candle = data.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range > 0:
                # Pin bar detection
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                
                if upper_wick > 2 * body_size:
                    actions.append({
                        'type': 'bearish_pin_bar',
                        'index': i,
                        'strength': min(upper_wick / body_size / 5, 1.0)
                    })
                elif lower_wick > 2 * body_size:
                    actions.append({
                        'type': 'bullish_pin_bar',
                        'index': i,
                        'strength': min(lower_wick / body_size / 5, 1.0)
                    })
        
        return actions

__all__ = ['ChartAnalyzer']
