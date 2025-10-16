from flask_sock import Sock

sock = Sock()

class Ws_controller:
    def __init__(self): 
        pass
        
    def register_ws_routes(self, app):
        sock.init_app(app)

        @sock.route('/ws/ai-video-stream')
        def websocket(ws):
            print("📦 Video Stream client connected")
            app.connected_ws_clients["video-stream"] = ws

           # self.manager.set_ws(ws)
            print("sockets: ", app.connected_ws_clients)
            try:
                while True:
                    msg = ws.receive()
                    if msg is None:
                        break
                    app.connected_ws_clients['video-stream'].send()
                    print("Object Info message from client:", msg)

            except Exception as e:
                print("Video Strean WebSocket error:", e)
            finally:
                print("Videa stream Client disconnected")
                app.connected_ws_clients.pop("video-stream")


        @sock.route('/ws/object-info')
        def object_info_websocket(ws):
            print("📦 Object info client connected")
            app.connected_ws_clients["object-info"] = ws
            try:
                while True:
                    msg = ws.receive()
                    if msg is None:
                        break
                    app.connected_ws_clients['object-info'].send()
                    print("Object Info message from client:", msg)
            except Exception as e:
                print("Object Info WS error:", e)
            finally:
                print("📦 Client disconnected from object info")
                app.connected_ws_clients.pop("object-info")
