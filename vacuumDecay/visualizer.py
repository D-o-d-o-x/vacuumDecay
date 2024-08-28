import threading
import time
import networkx as nx
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from io import BytesIO
import base64

class Visualizer:
    def __init__(self, runtime):
        self.runtime = runtime
        self.graph = nx.DiGraph()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.init_flask()
        self.update_thread = threading.Thread(target=self.update_periodically)
        self.update_thread.daemon = True
        self.update_thread.start()

    def init_flask(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/data')
        def data():
            return jsonify(self.get_data())

        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')

    def send_update(self):
        self.socketio.emit('update', self.get_data())

    def update_periodically(self):
        while True:
            self.send_update()
            time.sleep(1)

    def run(self):
        self.socketio.run(self.app, debug=True, use_reloader=False)

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def get_data(self):
        nodes_data = []
        edges_data = []

        def add_node_data(node, depth=0):
            img = None
            if node.state.getImage():  # depth <= 2:
                buffered = BytesIO()
                node.state.getImage().save(buffered, format="JPEG")
                img = base64.b64encode(buffered.getvalue()).decode("utf-8")

            nodes_data.append({
                'id': id(node),
                'parentId': id(node.parent) if node.parent else None,
                'image': img,
                'currentPlayer': node.state.curPlayer,
                'winProbs': [node.getStrongFor(i) for i in range(node.state.playersNum)],
                'last_updated': node.last_updated
            })

            for child in node.childs:
                edges_data.append({'source': id(node), 'target': id(child)})
                add_node_data(child, depth=depth + 1)

        head_node = self.runtime.head
        if head_node:
            add_node_data(head_node)

        return {'nodes': nodes_data, 'edges': edges_data}
