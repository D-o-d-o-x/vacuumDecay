import threading
import time
import networkx as nx
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

class Visualizer:
    def __init__(self, universe):
        self.universe = universe
        self.graph = nx.DiGraph()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.init_flask()

    def init_flask(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/data')
        def data():
            nodes_data = []
            edges_data = []
            for node in self.universe.iter():
                nodes_data.append({
                    'id': id(node),
                    'image': node.state.getImage().tobytes() if node.state.getImage() else None,
                    'value': node.getScoreFor(node.state.curPlayer),
                    'last_updated': node.last_updated
                })
                for child in node.childs:
                    edges_data.append({'source': id(node), 'target': id(child)})
            return jsonify(nodes=nodes_data, edges=edges_data)

        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')

    def send_update(self):
        nodes_data = []
        edges_data = []
        for node in self.universe.iter():
            nodes_data.append({
                'id': id(node),
                'image': node.state.getImage().tobytes() if node.state.getImage() else None,
                'value': node.getScoreFor(node.state.curPlayer),
                'last_updated': node.last_updated
            })
            for child in node.childs:
                edges_data.append({'source': id(node), 'target': id(child)})
        self.socketio.emit('update', {'nodes': nodes_data, 'edges': edges_data})

    def run(self):
        self.socketio.run(self.app, debug=True, use_reloader=False)

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
