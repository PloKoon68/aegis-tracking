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

image_processing_service = None  # kaldırıldı, artık port bazlı olacak

class StreamReceiverService:
    def __init__(self, app): 
        self.app = app
        self.receivers = {}
        self.frame_queues = {}
        self.active_ports = set()
        self.stop_conditions = {}         # {port: bool}
        self.image_services = {}          # {port: ImageProcessingService()}


    def monitor_ports(self):
        while True:
            try:
                response = requests.get("http://localhost:8091/api/streams/list")
                if response.status_code == 200:
                    port_list = response.json()  # örn: [8551, 8552, 8553]
                    new_ports = set(port_list) - self.active_ports
                    removed_ports = self.active_ports - set(port_list)
                    
                    for port in removed_ports:
                        print(f"🧹 RTSP {port} kaldırıldı, thread durdurulacak.")
                        self.active_ports.remove(port)

                    for port in new_ports:
                        self.start_receiver(port)
                        self.active_ports.add(port)

                else:
                    print(f"⛔ Port listesi alınamadı: {response.status_code}")

            except Exception as e:
                print(f"⛔ Port kontrol hatası: {e}")

            time.sleep(3)  # 3 saniyede bir kontrol

    def start_receiver(self, rtsp_port):
        existing_thread = self.receivers.get(rtsp_port)

        if existing_thread and existing_thread.is_alive():
            print(f"⚠️ RTSP {rtsp_port} zaten dinleniyor.", flush=True)
            return

        self.frame_queues[rtsp_port] = Queue(maxsize=5)
        self.stop_conditions[rtsp_port] = False
        self.image_services[rtsp_port] = ImageProcessingService()

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

        while True:
            self.stop_conditions[rtsp_port] = False

            if rtsp_port not in self.active_ports:
                print(f"🛑 RTSP {rtsp_port} artık aktif değil, thread sonlandırılıyor.")
                break

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
            image_thread = threading.Thread(target=self.image_processing_thread, args=(rtsp_port,), daemon=True)
            image_thread.start()

            while True:
                if rtsp_port not in self.active_ports:
                    print(f"🛑 RTSP {rtsp_port} yayını durduruldu, bağlantı kesiliyor.")
                    break

                ret, frame = cap.read()
                if ret:
                    last_frame_ts = time.time()
                    if frame_queue.full():
                        frame_queue.get_nowait()
                    frame_queue.put_nowait(frame)
                else:
                    if time.time() - last_frame_ts > 10:
                        print(f"⏱️ {rtsp_port}: 10 sn frame yok, yeniden bağlanılıyor.")
                        self.stop_conditions[rtsp_port] = True
                        break

            cap.release()




    
    def image_processing_thread(self, udp_port):
        frame_queue = self.frame_queues[udp_port]
        image_service = self.image_services[udp_port]

        while True:
            if self.stop_conditions.get(udp_port, False):
                break

            frame = frame_queue.get()
            print(f"✅ [RTSP {udp_port}] Frame successfully reassembled and decoded!", flush=True)
            processed_frame, obj_info = image_service.process_frame(frame, udp_port)

            video_stream_ws = self.app.connected_ws_clients.get("video-stream")

            if udp_port == self.app.current_streaming_cam and video_stream_ws:
                try:
                    video_stream_ws.send(processed_frame)
                except Exception as e:
                    print(f"WebSocket send error: {e}", flush=True)

            object_info_ws = self.app.connected_ws_clients.get("object-info")
            if object_info_ws:
                try:
                    print("info: ", obj_info, flush=True)
                    object_info_ws.send(json.dumps(obj_info))
                except Exception as e:
                    print(f"WebSocket send error: {e}", flush=True)

    