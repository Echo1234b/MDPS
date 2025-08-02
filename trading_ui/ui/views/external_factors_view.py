from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QTabWidget, QLabel, QTableWidget, QPushButton,
                           QComboBox, QLineEdit)
from ..widgets.charts.price_chart import PriceChart

class ExternalFactorsView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_external_monitors()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # News & Economics Tab
        self.create_news_economics_tab()

        # Sentiment Metrics Tab
        self.create_sentiment_tab()

        # Blockchain & On-chain Tab
        self.create_blockchain_tab()

    def create_news_economics_tab(self):
        """Create news and economics monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # News filter controls
        filter_layout = QHBoxLayout()
        
        self.news_source = QComboBox()
        self.news_source.addItems(['All Sources', 'Reuters', 'Bloomberg', 'CNBC'])
        
        self.news_search = QLineEdit()
        self.news_search.setPlaceholderText('Search news...')
        
        self.refresh_news = QPushButton('Refresh')
        
        filter_layout.addWidget(QLabel('Source:'))
        filter_layout.addWidget(self.news_source)
        filter_layout.addWidget(self.news_search)
        filter_layout.addWidget(self.refresh_news)
        
        layout.addLayout(filter_layout)

        # News table
        self.news_table = QTableWidget()
        self.news_table.setColumnCount(4)
        self.news_table.setHorizontalHeaderLabels(['Time', 'Source', 'Title', 'Impact'])
        layout.addWidget(self.news_table)

        # Economic events
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(5)
        self.events_table.setHorizontalHeaderLabels(['Time', 'Event', 'Country', 'Actual', 'Forecast'])
        layout.addWidget(self.events_table)

        self.tab_widget.addTab(tab, "News & Economics")

    def create_sentiment_tab(self):
        """Create sentiment metrics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Sentiment indicators
        sentiment_layout = QHBoxLayout()
        
        self.fear_greed = self.create_sentiment_widget("Fear & Greed Index")
        self.twitter_sentiment = self.create_sentiment_widget("Twitter Sentiment")
        self.funding_rates = self.create_sentiment_widget("Funding Rates")
        
        sentiment_layout.addWidget(self.fear_greed)
        sentiment_layout.addWidget(self.twitter_sentiment)
        sentiment_layout.addWidget(self.funding_rates)
        
        layout.addLayout(sentiment_layout)

        # Sentiment chart
        self.sentiment_chart = PriceChart()
        layout.addWidget(self.sentiment_chart)

        self.tab_widget.addTab(tab, "Sentiment Metrics")

    def create_blockchain_tab(self):
        """Create blockchain and on-chain metrics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Blockchain metrics
        metrics_layout = QHBoxLayout()
        
        self.hashrate = self.create_metric_widget("Hashrate")
        self.whale_activity = self.create_metric_widget("Whale Activity")
        self.network_value = self.create_metric_widget("Network Value")
        
        metrics_layout.addWidget(self.hashrate)
        metrics_layout.addWidget(self.whale_activity)
        metrics_layout.addWidget(self.network_value)
        
        layout.addLayout(metrics_layout)

        # On-chain data table
        self.onchain_table = QTableWidget()
        self.onchain_table.setColumnCount(4)
        self.onchain_table.setHorizontalHeaderLabels(['Metric', 'Value', 'Change', 'Trend'])
        layout.addWidget(self.onchain_table)

        self.tab_widget.addTab(tab, "Blockchain & On-chain")

    def create_sentiment_widget(self, label):
        """Create sentiment monitoring widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label_widget = QLabel(label)
        value_label = QLabel("0")
        trend_label = QLabel("→")
        
        layout.addWidget(label_widget)
        layout.addWidget(value_label)
        layout.addWidget(trend_label)
        
        return widget

    def create_metric_widget(self, label):
        """Create metric monitoring widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label_widget = QLabel(label)
        value_label = QLabel("0")
        change_label = QLabel("+0.0%")
        
        layout.addWidget(label_widget)
        layout.addWidget(value_label)
        layout.addWidget(change_label)
        
        return widget

    def setup_external_monitors(self):
        """Setup external data monitoring connections"""
        self.event_system.register('news_update', self.update_news)
        self.event_system.register('economic_events_update', self.update_economic_events)
        self.event_system.register('sentiment_update', self.update_sentiment)
        self.event_system.register('blockchain_update', self.update_blockchain)

        # Connect signals
        self.refresh_news.clicked.connect(self.refresh_news_data)
        self.news_search.textChanged.connect(self.filter_news)

    def update_news(self, news_data):
        """Update news display"""
        self.news_table.setRowCount(len(news_data))
        for i, news in enumerate(news_data):
            self.news_table.setItem(i, 0, QTableWidgetItem(news['time']))
            self.news_table.setItem(i, 1, QTableWidgetItem(news['source']))
            self.news_table.setItem(i, 2, QTableWidgetItem(news['title']))
            self.news_table.setItem(i, 3, QTableWidgetItem(news['impact']))

    def update_economic_events(self, events_data):
        """Update economic events display"""
        self.events_table.setRowCount(len(events_data))
        for i, event in enumerate(events_data):
            self.events_table.setItem(i, 0, QTableWidgetItem(event['time']))
            self.events_table.setItem(i, 1, QTableWidgetItem(event['event']))
            self.events_table.setItem(i, 2, QTableWidgetItem(event['country']))
            self.events_table.setItem(i, 3, QTableWidgetItem(str(event['actual'])))
            self.events_table.setItem(i, 4, QTableWidgetItem(str(event['forecast'])))

    def update_sentiment(self, sentiment_data):
        """Update sentiment metrics display"""
        # Update fear & greed index
        fg_value = self.fear_greed.findChildren(QLabel)[1]
        fg_trend = self.fear_greed.findChildren(QLabel)[2]
        fg_value.setText(str(sentiment_data['fear_greed']))
        fg_trend.setText(sentiment_data['fg_trend'])
        
        # Update twitter sentiment
        tw_value = self.twitter_sentiment.findChildren(QLabel)[1]
        tw_trend = self.twitter_sentiment.findChildren(QLabel)[2]
        tw_value.setText(str(sentiment_data['twitter_sentiment']))
        tw_trend.setText(sentiment_data['tw_trend'])
        
        # Update funding rates
        fr_value = self.funding_rates.findChildren(QLabel)[1]
        fr_trend = self.funding_rates.findChildren(QLabel)[2]
        fr_value.setText(str(sentiment_data['funding_rates']))
        fr_trend.setText(sentiment_data['fr_trend'])
        
        # Update sentiment chart
        self.sentiment_chart.update_data(sentiment_data['chart_data'])

    def update_blockchain(self, blockchain_data):
        """Update blockchain metrics display"""
        # Update hashrate
        hr_value = self.hashrate.findChildren(QLabel)[1]
        hr_change = self.hashrate.findChildren(QLabel)[2]
        hr_value.setText(str(blockchain_data['hashrate']))
        hr_change.setText(blockchain_data['hr_change'])
        
        # Update whale activity
        wa_value = self.whale_activity.findChildren(QLabel)[1]
        wa_change = self.whale_activity.findChildren(QLabel)[2]
        wa_value.setText(str(blockchain_data['whale_activity']))
        wa_change.setText(blockchain_data['wa_change'])
        
        # Update network value
        nv_value = self.network_value.findChildren(QLabel)[1]
        nv_change = self.network_value.findChildren(QLabel)[2]
        nv_value.setText(str(blockchain_data['network_value']))
        nv_change.setText(blockchain_data['nv_change'])
        
        # Update on-chain table
        self.onchain_table.setRowCount(len(blockchain_data['onchain_metrics']))
        for i, metric in enumerate(blockchain_data['onchain_metrics']):
            self.onchain_table.setItem(i, 0, QTableWidgetItem(metric['name']))
            self.onchain_table.setItem(i, 1, QTableWidgetItem(str(metric['value'])))
            self.onchain_table.setItem(i, 2, QTableWidgetItem(metric['change']))
            self.onchain_table.setItem(i, 3, QTableWidgetItem(metric['trend']))

    def refresh_news_data(self):
        """Refresh news data"""
        self.event_system.emit('refresh_news', {
            'source': self.news_source.currentText(),
            'search': self.news_search.text()
        })

    def filter_news(self):
        """Filter news based on search text"""
        search_text = self.news_search.text()
        self.event_system.emit('filter_news', {'search': search_text})
