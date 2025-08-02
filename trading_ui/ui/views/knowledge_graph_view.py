from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLineEdit, QComboBox, QLabel,
                           QGraphicsView, QGraphicsScene, QTableWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QBrush, QColor, QPainter

class KnowledgeGraphView(QWidget):
    def __init__(self, event_system):
        super().__init__()
        self.event_system = event_system
        self.init_ui()
        self.setup_graph_view()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Control panel
        control_panel = QHBoxLayout()
        
        self.pattern_selector = QComboBox()
        self.pattern_selector.addItems(['All Patterns', 'Candlestick', 'Harmonic', 'Elliott Wave'])
        
        self.event_filter = QLineEdit()
        self.event_filter.setPlaceholderText('Filter events...')
        
        self.search_button = QPushButton('Search')
        self.clear_button = QPushButton('Clear')
        
        control_panel.addWidget(QLabel('Pattern:'))
        control_panel.addWidget(self.pattern_selector)
        control_panel.addWidget(QLabel('Event Filter:'))
        control_panel.addWidget(self.event_filter)
        control_panel.addWidget(self.search_button)
        control_panel.addWidget(self.clear_button)
        
        layout.addLayout(control_panel)

        # Graph view
        self.graph_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graph_view.setScene(self.scene)
        layout.addWidget(self.graph_view)

        # Relationship details table
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(4)
        self.details_table.setHorizontalHeaderLabels(['Pattern', 'Event', 'Strength', 'Frequency'])
        layout.addWidget(self.details_table)

    def setup_graph_view(self):
        """Setup graph view properties"""
        self.graph_view.setRenderHint(QPainter.Antialiasing)
        self.graph_view.setDragMode(QGraphicsView.RubberBandDrag)
        self.graph_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

    def draw_graph(self, nodes, edges):
        """Draw knowledge graph"""
        self.scene.clear()
        
        # Draw nodes
        node_items = {}
        for node in nodes:
            item = self.scene.addEllipse(
                node['x'], node['y'], 60, 60,
                QPen(Qt.black, 2),
                QBrush(QColor(node['color']))
            )
            item.setFlag(QGraphicsItem.ItemIsMovable)
            node_items[node['id']] = item

        # Draw edges
        for edge in edges:
            self.scene.addLine(
                edge['x1'], edge['y1'],
                edge['x2'], edge['y2'],
                QPen(Qt.black, 1, Qt.DashLine)
            )

    def update_details_table(self, relationships):
        """Update relationship details table"""
        self.details_table.setRowCount(len(relationships))
        for i, rel in enumerate(relationships):
            self.details_table.setItem(i, 0, QTableWidgetItem(rel['pattern']))
            self.details_table.setItem(i, 1, QTableWidgetItem(rel['event']))
            self.details_table.setItem(i, 2, QTableWidgetItem(str(rel['strength'])))
            self.details_table.setItem(i, 3, QTableWidgetItem(str(rel['frequency'])))

    def setup_connections(self):
        """Setup signal connections"""
        self.search_button.clicked.connect(self.search_relationships)
        self.clear_button.clicked.connect(self.clear_view)
        self.event_system.register('graph_update', self.update_graph)

    def search_relationships(self):
        """Search for pattern-event relationships"""
        search_params = {
            'pattern': self.pattern_selector.currentText(),
            'event_filter': self.event_filter.text()
        }
        self.event_system.emit('search_relationships', search_params)

    def clear_view(self):
        """Clear graph view"""
        self.scene.clear()
        self.details_table.setRowCount(0)

    def update_graph(self, data):
        """Update graph with new data"""
        self.draw_graph(data['nodes'], data['edges'])
        self.update_details_table(data['relationships'])
