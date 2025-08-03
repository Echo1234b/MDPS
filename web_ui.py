#!/usr/bin/env python3
"""
MDPS Web UI - Simple web interface for the Market Data Processing System
Uses only standard library to avoid dependencies
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import __init__ as mdps_init
    from config import MDPSConfig
    HAS_MDPS = True
except ImportError as e:
    print(f"Could not import MDPS: {e}")
    HAS_MDPS = False

class MDPSWebHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MDPS web interface"""
    
    def __init__(self, *args, **kwargs):
        self.mdps_system = kwargs.pop('mdps_system', None)
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.serve_main_page()
        elif path == '/api/status':
            self.serve_status()
        elif path == '/api/run-cycle':
            self.serve_run_cycle()
        elif path == '/api/prediction':
            self.serve_prediction()
        elif path == '/styles.css':
            self.serve_css()
        elif path == '/script.js':
            self.serve_js()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        """Serve the main HTML page"""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MDPS - Market Data Processing System</title>
    <link rel="stylesheet" href="/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Market Data Processing System (MDPS)</h1>
            <p>Advanced AI-Powered Trading Analytics</p>
        </header>
        
        <div class="dashboard">
            <div class="card status-card">
                <h2>📊 System Status</h2>
                <div id="status-content">
                    <div class="loading">Loading...</div>
                </div>
            </div>
            
            <div class="card controls-card">
                <h2>🎮 Controls</h2>
                <button id="run-cycle-btn" class="btn btn-primary">Run Analysis Cycle</button>
                <button id="get-prediction-btn" class="btn btn-secondary">Get Prediction</button>
                <div class="symbols-input">
                    <label for="symbols">Symbols:</label>
                    <input type="text" id="symbols" value="EURUSD,GBPUSD,USDJPY" placeholder="Enter symbols">
                </div>
                <div class="timeframe-input">
                    <label for="timeframe">Timeframe:</label>
                    <select id="timeframe">
                        <option value="M1">1 Minute</option>
                        <option value="M5" selected>5 Minutes</option>
                        <option value="M15">15 Minutes</option>
                        <option value="H1">1 Hour</option>
                        <option value="H4">4 Hours</option>
                        <option value="D1">Daily</option>
                    </select>
                </div>
            </div>
            
            <div class="card results-card">
                <h2>📈 Analysis Results</h2>
                <div id="results-content">
                    <div class="no-data">No analysis run yet. Click "Run Analysis Cycle" to start.</div>
                </div>
            </div>
            
            <div class="card prediction-card">
                <h2>🔮 AI Prediction</h2>
                <div id="prediction-content">
                    <div class="no-data">No prediction available. Run analysis first.</div>
                </div>
            </div>
            
            <div class="card logs-card">
                <h2>📝 System Logs</h2>
                <div id="logs-content">
                    <div class="log-entry">System initialized</div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="/script.js"></script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_css(self):
        """Serve CSS styles"""
        css = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 30px;
    color: white;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

header p {
    font-size: 1.2rem;
    opacity: 0.9;
}

.dashboard {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}

.card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
}

.card h2 {
    margin-bottom: 15px;
    color: #2c3e50;
    font-size: 1.4rem;
}

.btn {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
    margin: 5px;
    transition: all 0.3s ease;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.btn-secondary {
    background: linear-gradient(45deg, #36d1dc, #5b86e5);
}

.symbols-input, .timeframe-input {
    margin: 10px 0;
}

.symbols-input input, .timeframe-input select {
    width: 100%;
    padding: 10px;
    border: 2px solid #e1e8ed;
    border-radius: 5px;
    font-size: 1rem;
    margin-top: 5px;
}

.loading, .no-data {
    text-align: center;
    color: #666;
    font-style: italic;
    padding: 20px;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 10px;
}

.status-item {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
}

.status-item .label {
    font-weight: bold;
    color: #666;
    font-size: 0.9rem;
}

.status-item .value {
    font-size: 1.5rem;
    margin-top: 5px;
    color: #2c3e50;
}

.signal-buy { color: #27ae60; }
.signal-sell { color: #e74c3c; }
.signal-hold { color: #f39c12; }

.prediction-details {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-top: 10px;
}

.prediction-item {
    background: #ecf0f1;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
}

.confidence-high { background-color: #d5f4e6; }
.confidence-medium { background-color: #fef9e7; }
.confidence-low { background-color: #fadbd8; }

.log-entry {
    padding: 8px;
    border-left: 3px solid #3498db;
    margin: 5px 0;
    background: #f8f9fa;
    font-family: monospace;
    font-size: 0.9rem;
}

.log-entry.error {
    border-left-color: #e74c3c;
    background: #fdedec;
}

.log-entry.success {
    border-left-color: #27ae60;
    background: #eafaf1;
}

@media (max-width: 768px) {
    .dashboard {
        grid-template-columns: 1fr;
    }
    
    header h1 {
        font-size: 2rem;
    }
    
    .btn {
        display: block;
        width: 100%;
        margin: 5px 0;
    }
}
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/css')
        self.end_headers()
        self.wfile.write(css.encode('utf-8'))
    
    def serve_js(self):
        """Serve JavaScript"""
        js = """
class MDPSInterface {
    constructor() {
        this.setupEventListeners();
        this.loadStatus();
        this.startPeriodicUpdates();
    }
    
    setupEventListeners() {
        document.getElementById('run-cycle-btn').addEventListener('click', () => {
            this.runAnalysisCycle();
        });
        
        document.getElementById('get-prediction-btn').addEventListener('click', () => {
            this.getPrediction();
        });
    }
    
    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            this.updateStatusDisplay(data);
        } catch (error) {
            console.error('Error loading status:', error);
            this.addLogEntry('Error loading system status', 'error');
        }
    }
    
    async runAnalysisCycle() {
        const btn = document.getElementById('run-cycle-btn');
        const originalText = btn.textContent;
        btn.textContent = 'Running...';
        btn.disabled = true;
        
        try {
            const symbols = document.getElementById('symbols').value;
            const timeframe = document.getElementById('timeframe').value;
            
            const response = await fetch(`/api/run-cycle?symbols=${symbols}&timeframe=${timeframe}`);
            const data = await response.json();
            
            this.updateResultsDisplay(data);
            this.addLogEntry(`Analysis cycle completed - Signal: ${data.signals?.signal || 'none'}`, 'success');
        } catch (error) {
            console.error('Error running cycle:', error);
            this.addLogEntry('Error running analysis cycle', 'error');
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }
    
    async getPrediction() {
        const btn = document.getElementById('get-prediction-btn');
        const originalText = btn.textContent;
        btn.textContent = 'Processing...';
        btn.disabled = true;
        
        try {
            const response = await fetch('/api/prediction');
            const data = await response.json();
            
            this.updatePredictionDisplay(data);
            this.addLogEntry(`Prediction generated: ${data.direction} (${(data.confidence * 100).toFixed(1)}% confidence)`, 'success');
        } catch (error) {
            console.error('Error getting prediction:', error);
            this.addLogEntry('Error getting prediction', 'error');
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }
    
    updateStatusDisplay(data) {
        const content = document.getElementById('status-content');
        content.innerHTML = `
            <div class="status-grid">
                <div class="status-item">
                    <div class="label">System Status</div>
                    <div class="value">${data.status}</div>
                </div>
                <div class="status-item">
                    <div class="label">Models Loaded</div>
                    <div class="value">${data.models_loaded ? 'Yes' : 'No'}</div>
                </div>
                <div class="status-item">
                    <div class="label">Last Update</div>
                    <div class="value">${data.last_update}</div>
                </div>
                <div class="status-item">
                    <div class="label">Uptime</div>
                    <div class="value">${data.uptime}</div>
                </div>
            </div>
        `;
    }
    
    updateResultsDisplay(data) {
        const content = document.getElementById('results-content');
        const signal = data.signals?.signal || 'none';
        const signalClass = `signal-${signal}`;
        
        content.innerHTML = `
            <div class="status-grid">
                <div class="status-item">
                    <div class="label">Trading Signal</div>
                    <div class="value ${signalClass}">${signal.toUpperCase()}</div>
                </div>
                <div class="status-item">
                    <div class="label">Confidence</div>
                    <div class="value">${((data.predictions?.confidence || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="status-item">
                    <div class="label">Patterns Found</div>
                    <div class="value">${data.patterns?.patterns?.length || 0}</div>
                </div>
                <div class="status-item">
                    <div class="label">Market Trend</div>
                    <div class="value">${data.market_context?.trend || 'Unknown'}</div>
                </div>
            </div>
            <div style="margin-top: 15px;">
                <strong>Entry Price:</strong> ${data.signals?.entry_price?.toFixed(4) || 'N/A'}<br>
                <strong>Stop Loss:</strong> ${data.signals?.stop_loss?.toFixed(4) || 'N/A'}<br>
                <strong>Take Profit:</strong> ${data.signals?.take_profit?.toFixed(4) || 'N/A'}<br>
                <strong>Risk/Reward:</strong> ${data.signals?.risk_reward_ratio?.toFixed(2) || 'N/A'}
            </div>
        `;
    }
    
    updatePredictionDisplay(data) {
        const content = document.getElementById('prediction-content');
        const confidence = data.confidence || 0;
        const confidenceClass = confidence > 0.8 ? 'confidence-high' : 
                               confidence > 0.6 ? 'confidence-medium' : 'confidence-low';
        
        content.innerHTML = `
            <div class="prediction-details">
                <div class="prediction-item ${confidenceClass}">
                    <div class="label">Direction</div>
                    <div class="value signal-${data.direction}">${data.direction?.toUpperCase() || 'UNKNOWN'}</div>
                </div>
                <div class="prediction-item">
                    <div class="label">Confidence</div>
                    <div class="value">${(confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="prediction-item">
                    <div class="label">Target</div>
                    <div class="value">${((data.target || 0) * 100).toFixed(2)}%</div>
                </div>
                <div class="prediction-item">
                    <div class="label">Stop Loss</div>
                    <div class="value">${((data.stop_loss || 0) * 100).toFixed(2)}%</div>
                </div>
            </div>
            <div style="margin-top: 15px;">
                <strong>Model:</strong> ${data.model_ensemble || 'Unknown'}<br>
                <strong>Features Used:</strong> ${data.features_used || 'Unknown'}<br>
                <strong>Market Regime:</strong> ${data.market_regime || 'Unknown'}
            </div>
        `;
    }
    
    addLogEntry(message, type = 'info') {
        const logsContent = document.getElementById('logs-content');
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${timestamp}] ${message}`;
        
        logsContent.insertBefore(entry, logsContent.firstChild);
        
        // Keep only last 10 entries
        const entries = logsContent.children;
        if (entries.length > 10) {
            logsContent.removeChild(entries[entries.length - 1]);
        }
    }
    
    startPeriodicUpdates() {
        // Update status every 30 seconds
        setInterval(() => {
            this.loadStatus();
        }, 30000);
    }
}

// Initialize the interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MDPSInterface();
});
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'application/javascript')
        self.end_headers()
        self.wfile.write(js.encode('utf-8'))
    
    def serve_status(self):
        """Serve system status as JSON"""
        status = {
            'status': 'Online',
            'models_loaded': True,
            'last_update': datetime.now().strftime('%H:%M:%S'),
            'uptime': '00:15:30'  # Mock uptime
        }
        
        self.send_json_response(status)
    
    def serve_run_cycle(self):
        """Run an MDPS analysis cycle"""
        query_params = parse_qs(urlparse(self.path).query)
        symbols = query_params.get('symbols', ['EURUSD,GBPUSD,USDJPY'])[0].split(',')
        timeframe = query_params.get('timeframe', ['M5'])[0]
        
        if HAS_MDPS and hasattr(self, 'mdps_system') and self.mdps_system:
            try:
                result = self.mdps_system.run_single_cycle(symbols, timeframe)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({'error': str(e), 'success': False})
        else:
            # Mock response
            mock_result = {
                'success': True,
                'signals': {
                    'signal': 'buy',
                    'strength': 0.75,
                    'entry_price': 1.0850,
                    'stop_loss': 1.0820,
                    'take_profit': 1.0920,
                    'risk_reward_ratio': 2.33
                },
                'predictions': {
                    'direction': 'buy',
                    'confidence': 0.78,
                    'target': 0.025,
                    'stop_loss': 0.015
                },
                'patterns': {
                    'patterns': [{'type': 'double_bottom', 'confidence': 0.8}]
                },
                'market_context': {
                    'trend': 'uptrend',
                    'volatility': 'normal',
                    'regime': 'trending'
                }
            }
            self.send_json_response(mock_result)
    
    def serve_prediction(self):
        """Serve AI prediction"""
        if HAS_MDPS and hasattr(self, 'mdps_system') and self.mdps_system:
            # Would call actual prediction system
            pass
        
        # Mock prediction
        mock_prediction = {
            'direction': 'sell',
            'confidence': 0.82,
            'target': 0.028,
            'stop_loss': 0.012,
            'model_ensemble': 'ensemble_6_models',
            'features_used': 45,
            'market_regime': 'volatile'
        }
        self.send_json_response(mock_prediction)
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        pass

class MDPSWebServer:
    """MDPS Web Server"""
    
    def __init__(self, port=8080):
        self.port = port
        self.server = None
        self.mdps_system = None
        
        # Initialize MDPS system if available
        if HAS_MDPS:
            try:
                config = MDPSConfig()
                self.mdps_system = self._create_mdps_system(config)
            except Exception as e:
                logging.error(f"Failed to initialize MDPS system: {e}")
    
    def _create_mdps_system(self, config):
        """Create MDPS system"""
        class SimpleMDPS:
            def __init__(self, config):
                self.config = config
                self.data_collector = mdps_init.DataCollector(config)
                self.data_cleaner = mdps_init.DataCleaner(config)
                self.feature_engine = mdps_init.FeatureEngine(config)
                self.chart_analyzer = mdps_init.ChartAnalyzer(config)
                self.market_analyzer = mdps_init.MarketAnalyzer(config)
                self.external_factors = mdps_init.ExternalFactors(config)
                self.prediction_engine = mdps_init.PredictionEngine(config)
                self.strategy_manager = mdps_init.StrategyManager(config)
            
            def run_single_cycle(self, symbols, timeframe):
                """Run a single processing cycle"""
                try:
                    # Collect and process data
                    raw_data = self.data_collector.collect_data(symbols, timeframe)
                    clean_data = self.data_cleaner.process(raw_data)
                    features = self.feature_engine.generate_features(clean_data)
                    chart_patterns = self.chart_analyzer.analyze(clean_data)
                    market_context = self.market_analyzer.analyze_structure(clean_data)
                    external_data = self.external_factors.get_current_factors()
                    predictions = self.prediction_engine.predict(features, chart_patterns, market_context, external_data)
                    signals = self.strategy_manager.execute_decisions(predictions, market_context, external_data)
                    
                    return {
                        'success': True,
                        'signals': signals,
                        'predictions': predictions,
                        'patterns': chart_patterns,
                        'market_context': market_context
                    }
                except Exception as e:
                    return {'success': False, 'error': str(e)}
        
        return SimpleMDPS(config)
    
    def create_handler(self):
        """Create request handler with MDPS system"""
        mdps_system = self.mdps_system
        
        class Handler(MDPSWebHandler):
            def __init__(self, *args, **kwargs):
                self.mdps_system = mdps_system
                super().__init__(*args, **kwargs)
        
        return Handler
    
    def start(self):
        """Start the web server"""
        try:
            handler_class = self.create_handler()
            self.server = HTTPServer(('localhost', self.port), handler_class)
            
            print(f"🌐 MDPS Web Interface starting on http://localhost:{self.port}")
            print("🚀 Open your browser and navigate to the URL above")
            print("⏹️  Press Ctrl+C to stop the server")
            
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
            self.stop()
        except Exception as e:
            print(f"❌ Server error: {e}")
    
    def stop(self):
        """Stop the web server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

def main():
    """Main entry point"""
    print("🚀 MDPS Web Interface")
    print("=" * 50)
    
    server = MDPSWebServer(port=8080)
    server.start()

if __name__ == "__main__":
    main()