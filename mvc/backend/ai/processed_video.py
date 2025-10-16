import asyncio
import websockets
import cv2
import numpy as np
import requests as http

YOLO_WS_URL = "ws://localhost:4000/ws/ai-video-stream"

async def receive_video():
    async with websockets.connect(YOLO_WS_URL, max_size=None) as websocket:
        print("✅ WebSocket bağlantısı kuruldu.")
        http.post("http://localhost:4000/video/8552")

        while True:
            try:
                # WebSocket'ten blob verisini al
                data = await websocket.recv()

                # Byte verisini numpy array'e çevir
                np_data = np.frombuffer(data, dtype=np.uint8)

                # Görüntüyü decode et
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Görüntüyü ekranda göster
                    cv2.imshow("YOLO Video Stream", frame)

                    # 'q' tuşuna basıldığında çık
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("⚠️ Görüntü decode edilemedi.")
            except Exception as e:
                print("🚨 Hata:", e)

        cv2.destroyAllWindows()

asyncio.run(receive_video())
