from flask import Flask
from flask_cors import CORS
from config import Config

from controller.coordinates_controller import coordinates_bp
from controller.ws_controller import register_ws_routes
from controller.heartbeat_controller import heartbeat_sender, hearbeat_bp


import threading


from model.coordinates_model import db  # db nesnesini içeri al

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

db.init_app(app)  # 🔥 SQLAlchemy'yi Flask ile bağla

app.connected_ws_clients = {}

app.register_blueprint(coordinates_bp)
app.register_blueprint(hearbeat_bp)

register_ws_routes(app)

@app.route('/')
def home():
    return 'Server running'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 🔧 Tabloları oluştur
    threading.Thread(target=heartbeat_sender, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, debug=True)
