import cv2
import socket
import threading
import time
import os
import av

class StreamManager:
    def __init__(self):
        self.active_streams = {}
        self.base_port = 4000

    def get_udp_port(self, video_id):
        return self.base_port + int(video_id)

    def start_stream(self, video_id):
        if video_id in self.active_streams:
            return f"🎥 Video {video_id} zaten yayında."

        stop_flag = threading.Event()
        t = threading.Thread(target=self.stream_video, args=(video_id, stop_flag))
        t.start()
        self.active_streams[video_id] = {"thread": t, "stop_flag": stop_flag}
        return f"✅ Video {video_id} yayına başladı."

    def stop_stream(self, video_id):
        if video_id not in self.active_streams:
            return f"⚠️ Video {video_id} yayında değil."

        self.active_streams[video_id]["stop_flag"].set()
        self.active_streams[video_id]["thread"].join()
        del self.active_streams[video_id]
        return f"⛔ Video {video_id} yayını durduruldu."

    def get_status(self):
        return {"active_streams": list(self.active_streams.keys())}

    def stream_video(self, video_id, stop_flag):
        path = os.path.join(os.path.dirname(__file__), "videos", f"{video_id}.mp4")
        if not os.path.exists(path):
            print(f"❌ Dosya bulunamadı: {path}")
            return

        cap = cv2.VideoCapture(path)
        udp_ip = "127.0.0.1"
        #udp_ip = "ai"
        udp_port = self.get_udp_port(video_id)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25
        frame_interval = 1 / fps

        output = av.open('dummy.h264', mode='w', format='h264')
        stream = output.add_stream("libx264", rate=int(fps))
        stream.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stream.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stream.pix_fmt = 'yuv420p'

        gönderilen_paket_sayısı = 0
        gönderilen_toplam_byte = 0
        CHUNK_SIZE = 6000

        while cap.isOpened() and not stop_flag.is_set():
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            packets = stream.encode(av_frame)

            for packet in packets:
                data = bytes(packet)
                for i in range(0, len(data), CHUNK_SIZE):
                    chunk = data[i:i + CHUNK_SIZE]
                    sock.sendto(chunk, (udp_ip, udp_port))
                    gönderilen_paket_sayısı += 1
                    gönderilen_toplam_byte += len(chunk)

            elapsed = time.time() - start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

            print(f"🛑 Gönderilen Paket Sayısı: {gönderilen_paket_sayısı}")
            print(f"📤 Gönderilen Toplam Veri: {gönderilen_toplam_byte} B ≈ {gönderilen_toplam_byte / 1024:.2f} KB")

        cap.release()
        sock.close()
        print(f"🔚 Video {video_id} yayını sonlandırıldı.")
