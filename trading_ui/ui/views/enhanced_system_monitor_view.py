"""Enhanced System Monitor View with MDPS Integration"""
import sys
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QGroupBox, QLabel, QProgressBar, QTextEdit, QTabWidget,
                           QTableWidget, QTableWidgetItem, QPushButton, QSplitter,
                           QListWidget, QListWidgetItem, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg
from .base_view import BaseView

class SystemMonitorThread(QThread):
    """Thread for monitoring system resources"""
    
    system_update = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cpu_history = []
        self.memory_history = []
        self.disk_history = []
        self.network_history = []
        
    def run(self):
        self.running = True
        previous_net_io = psutil.net_io_counters()
        
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_per_core = psutil.cpu_percent(percpu=True)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                # Disk usage
                disk = psutil.disk_usage('/')
                
                # Network I/O
                current_net_io = psutil.net_io_counters()
                net_sent_per_sec = current_net_io.bytes_sent - previous_net_io.bytes_sent
                net_recv_per_sec = current_net_io.bytes_recv - previous_net_io.bytes_recv
                previous_net_io = current_net_io
                
                # Process information
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        proc_info = proc.info
                        if proc_info['cpu_percent'] > 0 or proc_info['memory_percent'] > 0:
                            processes.append(proc_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Sort by CPU usage
                processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
                
                # Update history
                current_time = time.time()
                self.cpu_history.append((current_time, cpu_percent))
                self.memory_history.append((current_time, memory.percent))
                self.disk_history.append((current_time, disk.percent))
                self.network_history.append((current_time, net_sent_per_sec + net_recv_per_sec))
                
                # Keep only last 100 points
                self.cpu_history = self.cpu_history[-100:]
                self.memory_history = self.memory_history[-100:]
                self.disk_history = self.disk_history[-100:]
                self.network_history = self.network_history[-100:]
                
                system_data = {
                    'cpu': {
                        'total': cpu_percent,
                        'per_core': cpu_per_core,
                        'history': self.cpu_history.copy()
                    },
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'used': memory.used,
                        'percent': memory.percent,
                        'history': self.memory_history.copy()
                    },
                    'disk': {
                        'total': disk.total,
                        'used': disk.used,
                        'free': disk.free,
                        'percent': disk.percent,
                        'history': self.disk_history.copy()
                    },
                    'network': {
                        'sent_per_sec': net_sent_per_sec,
                        'recv_per_sec': net_recv_per_sec,
                        'history': self.network_history.copy()
                    },
                    'processes': processes[:10],  # Top 10 processes
                    'timestamp': current_time
                }
                
                self.system_update.emit(system_data)
                
            except Exception as e:
                print(f"System monitor error: {e}")
            
            self.msleep(1000)  # Update every second
    
    def stop(self):
        self.running = False

class EnhancedSystemMonitorView(BaseView):
    """Enhanced system monitor view with comprehensive monitoring"""
    
    def __init__(self, event_system):
        super().__init__(event_system)
        self.monitor_thread = None
        self.mdps_status = {}
        self.start_time = datetime.now()
        self.init_ui()
        self.setup_monitoring()
        
    def init_ui(self):
        """Initialize the enhanced UI"""
        layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel - System resources
        left_panel = self.create_system_resources_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - MDPS status and processes
        right_panel = self.create_mdps_status_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter sizes (60% left, 40% right)
        main_splitter.setSizes([600, 400])
        
    def create_system_resources_panel(self):
        """Create system resources monitoring panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # System overview
        overview_group = QGroupBox("System Overview")
        overview_layout = QGridLayout()
        
        # CPU section
        cpu_label = QLabel("CPU Usage:")
        cpu_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
        """)
        self.cpu_label_text = QLabel("0%")
        
        # Memory section
        memory_label = QLabel("Memory Usage:")
        memory_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.memory_progress = QProgressBar()
        self.memory_progress.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #F7931E;
            }
        """)
        self.memory_label_text = QLabel("0 MB / 0 MB")
        
        # Disk section
        disk_label = QLabel("Disk Usage:")
        disk_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.disk_progress = QProgressBar()
        self.disk_progress.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #8CC152;
            }
        """)
        self.disk_label_text = QLabel("0 GB / 0 GB")
        
        # Network section
        network_label = QLabel("Network I/O:")
        network_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.network_label_text = QLabel("↑ 0 KB/s ↓ 0 KB/s")
        
        # Add to layout
        overview_layout.addWidget(cpu_label, 0, 0)
        overview_layout.addWidget(self.cpu_progress, 0, 1)
        overview_layout.addWidget(self.cpu_label_text, 0, 2)
        
        overview_layout.addWidget(memory_label, 1, 0)
        overview_layout.addWidget(self.memory_progress, 1, 1)
        overview_layout.addWidget(self.memory_label_text, 1, 2)
        
        overview_layout.addWidget(disk_label, 2, 0)
        overview_layout.addWidget(self.disk_progress, 2, 1)
        overview_layout.addWidget(self.disk_label_text, 2, 2)
        
        overview_layout.addWidget(network_label, 3, 0)
        overview_layout.addWidget(self.network_label_text, 3, 1, 1, 2)
        
        overview_group.setLayout(overview_layout)
        layout.addWidget(overview_group)
        
        # Real-time charts
        charts_group = QGroupBox("Performance Charts")
        charts_layout = QVBoxLayout()
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Usage (%)')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.addLegend()
        
        # Setup plots
        self.cpu_plot = self.plot_widget.plot(pen='c', name='CPU')
        self.memory_plot = self.plot_widget.plot(pen='m', name='Memory')
        self.disk_plot = self.plot_widget.plot(pen='g', name='Disk')
        
        charts_layout.addWidget(self.plot_widget)
        charts_group.setLayout(charts_layout)
        layout.addWidget(charts_group)
        
        # Process table
        process_group = QGroupBox("Top Processes")
        process_layout = QVBoxLayout()
        
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(4)
        self.process_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "Memory %"])
        self.process_table.horizontalHeader().setStretchLastSection(True)
        self.process_table.setMaximumHeight(200)
        
        process_layout.addWidget(self.process_table)
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        return widget
    
    def create_mdps_status_panel(self):
        """Create MDPS status monitoring panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # MDPS Status group
        mdps_group = QGroupBox("MDPS System Status")
        mdps_layout = QVBoxLayout()
        
        # Status indicators
        status_layout = QGridLayout()
        
        self.mdps_status_label = QLabel("Status: Not Running")
        self.mdps_status_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.mdps_uptime_label = QLabel("Uptime: 00:00:00")
        self.mdps_symbols_label = QLabel("Symbols: 0")
        self.mdps_predictions_label = QLabel("Predictions: 0")
        self.mdps_signals_label = QLabel("Signals: 0")
        
        status_layout.addWidget(QLabel("System Status:"), 0, 0)
        status_layout.addWidget(self.mdps_status_label, 0, 1)
        status_layout.addWidget(QLabel("Uptime:"), 1, 0)
        status_layout.addWidget(self.mdps_uptime_label, 1, 1)
        status_layout.addWidget(QLabel("Active Symbols:"), 2, 0)
        status_layout.addWidget(self.mdps_symbols_label, 2, 1)
        status_layout.addWidget(QLabel("Predictions/Hour:"), 3, 0)
        status_layout.addWidget(self.mdps_predictions_label, 3, 1)
        status_layout.addWidget(QLabel("Trading Signals:"), 4, 0)
        status_layout.addWidget(self.mdps_signals_label, 4, 1)
        
        mdps_layout.addLayout(status_layout)
        mdps_group.setLayout(mdps_layout)
        layout.addWidget(mdps_group)
        
        # Component status
        components_group = QGroupBox("Component Status")
        components_layout = QVBoxLayout()
        
        self.components_list = QListWidget()
        self.components_list.setMaximumHeight(150)
        
        # Initialize component list
        components = [
            "Data Collector",
            "Data Cleaner", 
            "Feature Engine",
            "Chart Analyzer",
            "Market Analyzer",
            "External Factors",
            "Prediction Engine",
            "Strategy Manager"
        ]
        
        for component in components:
            item = QListWidgetItem(f"● {component}: Stopped")
            item.setForeground(QColor("red"))
            self.components_list.addItem(item)
        
        components_layout.addWidget(self.components_list)
        components_group.setLayout(components_layout)
        layout.addWidget(components_group)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        
        # Create metrics display
        metrics_scroll = QScrollArea()
        metrics_widget = QWidget()
        metrics_widget_layout = QVBoxLayout(metrics_widget)
        
        self.processing_time_label = QLabel("Avg Processing Time: 0ms")
        self.accuracy_label = QLabel("Prediction Accuracy: 0%")
        self.success_rate_label = QLabel("Signal Success Rate: 0%")
        self.error_count_label = QLabel("Errors (24h): 0")
        
        metrics_widget_layout.addWidget(self.processing_time_label)
        metrics_widget_layout.addWidget(self.accuracy_label)
        metrics_widget_layout.addWidget(self.success_rate_label)
        metrics_widget_layout.addWidget(self.error_count_label)
        
        metrics_scroll.setWidget(metrics_widget)
        metrics_scroll.setMaximumHeight(120)
        metrics_layout.addWidget(metrics_scroll)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # System controls
        controls_group = QGroupBox("System Controls")
        controls_layout = QHBoxLayout()
        
        self.restart_button = QPushButton("Restart MDPS")
        self.restart_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        
        self.clear_logs_button = QPushButton("Clear Logs")
        self.export_metrics_button = QPushButton("Export Metrics")
        
        controls_layout.addWidget(self.restart_button)
        controls_layout.addWidget(self.clear_logs_button)
        controls_layout.addWidget(self.export_metrics_button)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        layout.addStretch()
        
        return widget
    
    def setup_monitoring(self):
        """Setup system monitoring"""
        # Start system monitor thread
        self.monitor_thread = SystemMonitorThread()
        self.monitor_thread.system_update.connect(self.update_system_display)
        self.monitor_thread.start()
        
        # Setup MDPS status timer
        self.mdps_timer = QTimer()
        self.mdps_timer.timeout.connect(self.update_mdps_status)
        self.mdps_timer.start(5000)  # Update every 5 seconds
    
    @pyqtSlot(dict)
    def update_system_display(self, data):
        """Update system resource displays"""
        try:
            # Update CPU
            cpu_percent = data['cpu']['total']
            self.cpu_progress.setValue(int(cpu_percent))
            self.cpu_label_text.setText(f"{cpu_percent:.1f}%")
            
            # Update Memory
            memory = data['memory']
            memory_percent = memory['percent']
            memory_used_gb = memory['used'] / (1024**3)
            memory_total_gb = memory['total'] / (1024**3)
            
            self.memory_progress.setValue(int(memory_percent))
            self.memory_label_text.setText(f"{memory_used_gb:.1f} GB / {memory_total_gb:.1f} GB")
            
            # Update Disk
            disk = data['disk']
            disk_percent = disk['percent']
            disk_used_gb = disk['used'] / (1024**3)
            disk_total_gb = disk['total'] / (1024**3)
            
            self.disk_progress.setValue(int(disk_percent))
            self.disk_label_text.setText(f"{disk_used_gb:.1f} GB / {disk_total_gb:.1f} GB")
            
            # Update Network
            network = data['network']
            sent_kb = network['sent_per_sec'] / 1024
            recv_kb = network['recv_per_sec'] / 1024
            self.network_label_text.setText(f"↑ {sent_kb:.1f} KB/s ↓ {recv_kb:.1f} KB/s")
            
            # Update charts
            self.update_charts(data)
            
            # Update process table
            self.update_process_table(data['processes'])
            
        except Exception as e:
            print(f"Error updating system display: {e}")
    
    def update_charts(self, data):
        """Update performance charts"""
        try:
            # Extract time and values for charts
            if data['cpu']['history']:
                cpu_times = [point[0] for point in data['cpu']['history']]
                cpu_values = [point[1] for point in data['cpu']['history']]
                
                memory_values = [point[1] for point in data['memory']['history']]
                disk_values = [point[1] for point in data['disk']['history']]
                
                # Convert timestamps to relative time (seconds from start)
                base_time = cpu_times[0] if cpu_times else 0
                relative_times = [(t - base_time) for t in cpu_times]
                
                # Update plots
                self.cpu_plot.setData(relative_times, cpu_values)
                self.memory_plot.setData(relative_times, memory_values)
                self.disk_plot.setData(relative_times, disk_values)
                
        except Exception as e:
            print(f"Error updating charts: {e}")
    
    def update_process_table(self, processes):
        """Update the process table"""
        try:
            self.process_table.setRowCount(len(processes))
            
            for i, proc in enumerate(processes):
                self.process_table.setItem(i, 0, QTableWidgetItem(str(proc['pid'])))
                self.process_table.setItem(i, 1, QTableWidgetItem(proc['name']))
                self.process_table.setItem(i, 2, QTableWidgetItem(f"{proc['cpu_percent']:.1f}%"))
                self.process_table.setItem(i, 3, QTableWidgetItem(f"{proc['memory_percent']:.1f}%"))
                
        except Exception as e:
            print(f"Error updating process table: {e}")
    
    def update_mdps_status(self):
        """Update MDPS system status"""
        # Calculate uptime
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        self.mdps_uptime_label.setText(f"Uptime: {uptime_str}")
        
        # TODO: Get real MDPS status from controller
        # For now, simulate status updates
        
    def update_data(self, data):
        """Update view with new MDPS data"""
        if data:
            # Update MDPS status from real data
            self.mdps_status_label.setText("Status: Running")
            self.mdps_status_label.setStyleSheet("color: green; font-weight: bold;")
            
            # Update component status
            for i in range(self.components_list.count()):
                item = self.components_list.item(i)
                text = item.text()
                component_name = text.split(':')[0]
                item.setText(f"{component_name}: Running")
                item.setForeground(QColor("green"))
            
            # Update metrics if available
            if 'predictions' in data:
                predictions = data['predictions']
                if hasattr(predictions, '__len__'):
                    self.mdps_predictions_label.setText(f"Predictions: {len(predictions)}")
            
            if 'signals' in data:
                signals = data['signals']
                if hasattr(signals, '__len__'):
                    self.mdps_signals_label.setText(f"Signals: {len(signals)}")
    
    def closeEvent(self, event):
        """Clean up when closing"""
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread.wait()
        event.accept()