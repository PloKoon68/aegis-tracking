from queue import Queue
import threading
import socket
import time
import av
import cv2
import json
from av import codec
import requests

from services.image_processing_service import ImageProcessingService

image_processing_service = ImageProcessingService()

class StreamReceiverService:
    def __init__(self, app): 
        self.app = app
        self.receivers = {}
        self.frame_queues = {}
        self.active_ports = set()

    def monitor_ports(self):
        while True:
            try:
                response = requests.get("http://localhost:8091/api/streams/list")
                if response.status_code == 200:
                    port_list = response.json()  # örn: [8551, 8552, 8553]
                    new_ports = set(port_list) - self.active_ports
                    print(new_ports)
                    

                    for port in new_ports:
                        self.start_receiver(port)
                        self.active_ports.add(port)

                else:
                    print(f"⛔ Port listesi alınamadı: {response.status_code}")
                time.sleep(3)

            except Exception as e:
                print(f"⛔ Port kontrol hatası: {e}")

            time.sleep(5)  # 5 saniyede bir kontrol

    def start_receiver(self, rtsp_port):
        if rtsp_port in self.receivers:
            print(f"⚠️ RTSP {rtsp_port} zaten dinleniyor.", flush=True)
            return

        self.frame_queues[rtsp_port] = Queue(maxsize=2)
        t = threading.Thread(target=self.receive_stream, args=(rtsp_port,))
        t.start()
        self.receivers[rtsp_port] = t
        print(f"✅ RTSP {rtsp_port} alıcı başlatıldı.", flush=True)

    def receive_stream(self, rtsp_port):
        reciver_thread = threading.Thread(target=self.receiver_thread, args=(rtsp_port,))

        reciver_thread.start()

    # def reciver_thread(self, udp_port):
    #     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    #     sock.bind(("0.0.0.0", udp_port))
    #     print(f"🎥 Dinleniyor: UDP {udp_port}")

    #     frame_queue = self.frame_queues[udp_port]

    #     buffer = b""
    #     paket_sayisi = 0
    #     alinan_toplam_byte = 0
    #     son_veri_zamani = time.time()
    #     CHUNK_SIZE = 60000

    #     codec = av.codec.CodecContext.create("h264", "r")

    #     while True:
    #         try:
    #             data, _ = sock.recvfrom(65536)
    #             buffer += data
    #             paket_sayisi += 1
    #             alinan_toplam_byte += len(data)

    #             now = time.time()
    #             if now - son_veri_zamani > 1:
    #                 print(f"⚠️ [UDP {udp_port}] 1 saniyedir veri alınmıyor!")
    #             son_veri_zamani = now

    #             if len(data) < CHUNK_SIZE:
    #                 try:
    #                     packets = codec.parse(buffer)
    #                     for packet in packets:
    #                         frames = codec.decode(packet)
    #                         for frame in frames:
    #                             img = frame.to_ndarray(format="bgr24")
    #                             #img = cv2.resize(img, (600, 300))
                                
    #                             frame_queue.put(img)

    #                     buffer = b""

    #                 except Exception as decode_error:
    #                     print(f"⛔ Decode hatası: {decode_error}")
    #                     buffer = b""

    #             print(f"📥 Alınan Paket Sayısı: {paket_sayisi}")
    #             print(f"📥 Alınan Toplam Veri: {alinan_toplam_byte} B ≈ {alinan_toplam_byte / 1024:.2f} KB")

    #         except Exception as e:
    #             print(f"⛔ Hata (UDP {udp_port}): {e}")
    #             break

    def receiver_thread(self, rtsp_port):
        rtsp_url = f"rtsp://10.152.16.187:{rtsp_port}/"
        frame_queue = self.frame_queues[rtsp_port]
        global stopCondition 

        while True:
            stopCondition = False
            # RTSP bağlantı denemesi (UDP + timeout)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            if not cap.isOpened():
                print(f"⛔ {rtsp_port} portunda açma hatası. 5 sn sonra yeniden deneniyor.")
                cap.release()
                time.sleep(5)
                continue

            print(f"📡 {rtsp_port} portunda bağlantı kuruldu.")
            last_frame_ts = time.time()
            image_thread = threading.Thread(target=self.image_processing_thread, args=(rtsp_port, ), daemon=True)
            image_thread.start()

            # Kare alma ve 10 sn kontrolü
            while True:
                ret, frame = cap.read()
                if ret:
                    last_frame_ts = time.time()
                    if frame_queue.full():
                        frame_queue.get_nowait()
                    frame_queue.put_nowait(frame)
                else:
                    if time.time() - last_frame_ts > 10:
                        print(f"⏱️ {rtsp_port}: 10 sn frame yok, yeniden bağlanılıyor.")
                        stopCondition = True
                        break

            cap.release()
            #time.sleep(5)  # 5 sn aralıkla yeniden bağlan


    
    def image_processing_thread(self, udp_port):
        frame_queue = self.frame_queues[udp_port]

        while True:
            if stopCondition:
                break
            frame = frame_queue.get()
            print(f"✅ [RTSP {udp_port}] Frame successfully reassembled and decoded!", flush=True)
            processed_frame, obj_info = image_processing_service.process_frame(frame, udp_port)
            print("obj returned: , ", obj_info)
            #obj_info['cam_id'] = udp_port

            """#to show the frames
            if processed_frame is not None:
                processed_frame = cv2.resize(processed_frame, (400, 200))
                cv2.imshow(f"UDP Stream {udp_port}", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            """

            video_stream_ws = self.app.connected_ws_clients.get("video-stream")
            print("vs: , ", video_stream_ws)

            if udp_port == self.app.current_streaming_cam and video_stream_ws:
                print("vs: , ", True)

                try:
                    video_stream_ws.send(processed_frame)
                except Exception as e:
                    print(f"WebSocket send error: {e}", flush=True)



            object_info_ws = self.app.connected_ws_clients.get("object-info")
            #onject info send
            if object_info_ws:
                try:
                    print("info: ", obj_info, flush=True)
                    object_info_ws.send(json.dumps(obj_info))
                except Exception as e:
                    print(f"WebSocket send error: {e}", flush=True)
    