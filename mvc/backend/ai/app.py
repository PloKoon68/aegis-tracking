import threading
from flask import Flask
from flask_cors import CORS

from controller.ws_controller import Ws_controller
from controller.video_request import video_bp

from services.stream_reciver_service import StreamReceiverService


app = Flask(__name__)
CORS(app)

app.register_blueprint(video_bp)

manager = StreamReceiverService(app)
    
def start_udp():
    # ports_to_listen = [8551, 8552, 8553, 8554, 8555]
    # for port in ports_to_listen:
    #     manager.start_receiver(port)
    threading.Thread(target=manager.monitor_ports, daemon=True).start()
    ws_controller = Ws_controller()
    ws_controller.register_ws_routes(app)

# Global list for connected WebSocket clients
app.connected_ws_clients = {}
app.current_streaming_cam = -1

# Register websocket routes
#register_ws_routes(app)

print("aldı: ", app.connected_ws_clients)
@app.route('/')
def home():
    print("bu da")
    return 'Server running'

if __name__ == '__main__':
    start_udp()
    app.run(host='0.0.0.0', port=4000, debug=True, use_reloader=False)
    