# from queue import Queue, Empty
# import queue
# import threading
# import time
# import cv2
# from flask import logging

# class Receiver:
#     def __init__(self): 
#         self.receivers = {}
#         self.frame_queues = {}

#     def start_receiver(self, rtsp_url):
#         if rtsp_url in self.receivers:
#             print(f"⚠️ RTSP {rtsp_url} zaten dinleniyor.", flush=True)
#             return

#         self.frame_queues[8553] = Queue()
#         t = threading.Thread(target=self.receive_stream, args=(rtsp_url,))
#         t.start()
#         self.receivers[rtsp_url] = t
#         print(f"✅ RTSP {rtsp_url} alıcı başlatıldı.", flush=True)

#     def receive_stream(self, rtsp_url):
#         reciver_thread = threading.Thread(target=self.reciver_thread, args=(rtsp_url,))

#         reciver_thread.start()


#     def reciver_thread(self, rtsp_url):
#         cap = cv2.VideoCapture(rtsp_url)

#         if not cap.isOpened():
#             logging.info(f"⛔ RTSP bağlantısı açılamadı: {rtsp_url}")
#             return

#         frame_queue = self.frame_queues[8553]

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logging.info(f"⚠️ RTSP akışından frame alınamadı: {rtsp_url}")
#                 time.sleep(0.5)
#                 continue
#             logging.info("alıyor")
#             frame_queue.put(frame)

#         cap.release()


import av
import cv2
import threading
import os
import time
from queue import Queue

class StreamReceiverService:
    def __init__(self, app=None):
        self.app = app
        self.receivers = {}       # rtsp_port -> thread
        self.frame_queues = {}    # rtsp_port -> Queue

    def start_receiver(self, rtsp_url, rtsp_port):
        if rtsp_port in self.receivers:
            print(f"⚠️ RTSP {rtsp_port} zaten dinleniyor.", flush=True)
            return

        self.frame_queues[rtsp_port] = Queue(maxsize=2)  # küçük kuyruk -> düşük gecikme
        t = threading.Thread(target=self.receive_stream, args=(rtsp_url, rtsp_port), daemon=True)
        t.start()
        self.receivers[rtsp_port] = t
        print(f"✅ RTSP {rtsp_url} alıcı başlatıldı.", flush=True)

    def receive_stream(self, rtsp_url, rtsp_port):
        threading.Thread(target=self.receiver_thread, args=(rtsp_url, rtsp_port), daemon=True).start()

    def receiver_thread(self, rtsp_url ,rtsp_port):
        rtsp_url = f"rtsp://10.152.16.187:{rtsp_port}/"
        frame_queue = self.frame_queues[rtsp_port]
        while True:
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
            threading.Thread(target=self.receiver_thread, args=(rtsp_url, rtsp_port), daemon=True).start()

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
                        break
                    time.sleep(0.1)

            cap.release()
            time.sleep(5)  # 5 sn aralıkla yeniden bağlan

    def image_processing_thread(self, rtsp_url, rtsp_port):
        frame_queue = self.frame_queues[rtsp_port]
        while True:
            frame = frame_queue.get()
            cv2.imshow(f"Stream - {rtsp_url}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

# Örnek kullanım
if __name__ == "__main__":
    service = StreamReceiverService()
    service.start_receiver("rtsp://10.152.16.187:8552", 8552)
    while True:
        time.sleep(1)






# import os
# os.environ["PATH"] = r"C:\\Users\\stj.mcirci\\ffmpeg\\bin" + os.environ["PATH"]

# import av
# import cv2

# rtsp_url = "rtsp://10.152.16.187:8553/video3"

# # TCP ile bağlanmayı zorla (VLC hangi protokolü kullanıyorsa onu seç)
# video = av.open(
#     rtsp_url,
#     options={
#         "rtsp_transport": "udp",   # veya "udp"
#         "stimeout": "5000000"      # 5 saniye timeout (mikrosaniye)
#     }
# )

# try:
#     for packet in video.demux():
#         if packet.stream.type == 'video':  # bytes yerine string kontrolü
#             for frame in packet.decode():
#                 img = frame.to_ndarray(format='bgr24')
#                 cv2.imshow("Test", img)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     raise KeyboardInterrupt
# except KeyboardInterrupt:
#     pass

# cv2.destroyAllWindows()

