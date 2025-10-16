from flask_sock import Sock

sock = Sock()

def register_ws_routes(app):
    sock.init_app(app)

    @sock.route('/ws/gps-data')
    def websocket(ws):
        print("Client connected")
        app.connected_ws_clients["gps-data"] = ws
        try:
            while True:
                msg = ws.receive()
                if msg is None:
                    break
                print("Client message:", msg)
                app.connected_ws_clients['gps-data'].send()
            
        except Exception as e:
            print("AI WebSocket error:", e)
        finally:
            print("Client disconnected")
            app.connected_ws_clients.pop("gps-data")
