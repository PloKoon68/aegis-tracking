from flask import Blueprint, request
import time
import requests

hearbeat_bp = Blueprint('hearbeat_bp', __name__)


@hearbeat_bp.route('/heartbeat', methods=['POST'])
def heartbeat():
    data = request.get_json()
    print(f"[{time.strftime('%H:%M:%S')}] Gelen Heartbeat: {data}")
    return {'message': 'Heartbeat alındı'}, 200

# Heartbeat background thread
def heartbeat_sender():
    while True:
        try:
            requests.post("http://localhost:8080/heartbeat", json={"status": "alive", "source": "flask-app"})
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Heartbeat hatası: {e}")
        time.sleep(1)