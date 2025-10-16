import datetime
import cv2
import os
import math
import requests
import torch
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, deque
from ultralytics import YOLO
from sklearn.preprocessing import MinMaxScaler

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== Constants =====
SEQ_LEN = 30
RISK_THRESHOLD = 75
PIXEL_TO_METER = 0.2
MPS_TO_KNOTS = 1.94384
FPS = 30
EARTH_RADIUS_KM = 6371.0

# ===== Model Paths =====
YOLO_PATH = "./services/aimodels/yolo11x.pt"
REGRESSOR_PATH = "./services/aimodels/model5.pkl"
TRANSFORMER_PATH = "./services/aimodels/transformer_model_weights.pth"

# ===== Load Models =====
yolo_model = YOLO(YOLO_PATH)
yolo_model.fuse()
yolo_model.to(device)  # YOLO'yu GPU'ya taşı

regression_model = joblib.load(REGRESSOR_PATH)  # CPU modeli

# ===== Transformer Model =====
class TimeSeriesTransformer(torch.nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1, max_seq_len=1000):
        super().__init__()
        self.embedding = torch.nn.Linear(input_dim, model_dim)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = torch.nn.LayerNorm(model_dim)
        self.regressor = torch.nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x[:, -1, :])
        return self.regressor(x)

model_time_series_transformer = TimeSeriesTransformer(17, 128, 4, 4).to(device)
model_time_series_transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device))
model_time_series_transformer.eval()

# ===== Buffer =====
buffer = defaultdict(lambda: deque(maxlen=SEQ_LEN))
risk_alarm_log = []

# ===== Helper Functions =====
def extract_displacement(dx, dy):
    return math.sqrt(dx**2 + dy**2)

def calculate_tti(x, y, dx, dy):
    pos = np.array([x, y])
    vel = np.array([dx, dy])
    distance = np.linalg.norm(pos)
    if distance == 0: return 0
    rel_speed = np.dot(pos, vel) / distance
    return float("inf") if rel_speed <= 0 else distance / rel_speed

def risk_from_tti(tti):
    if tti == float("inf"): return 0
    elif tti < 3: return 25
    elif tti < 6: return 15
    elif tti < 10: return 8
    else: return 2

def regression_risk_score(dx_true, dy_true, dx_pred, dy_pred):
    err = np.sqrt((dx_true - dx_pred)**2 + (dy_true - dy_pred)**2)
    return min(err * 10, 100)

def compute_features(det, obj_id, buffer, model5):
    x, y, w, h = det.xywh[0].tolist()
    frame_data = buffer[obj_id]
    dx = dy = 0

    if len(frame_data) > 0:
        prev_x, prev_y = frame_data[-1][3], frame_data[-1][4]
        dx = x - prev_x
        dy = y - prev_y

    displacement_px = extract_displacement(dx, dy)

    # --- VELOCITY (m/s ve knots) ---
    vx_mps = dx * PIXEL_TO_METER * FPS
    vy_mps = dy * PIXEL_TO_METER * FPS
    speed_mps = math.sqrt(vx_mps**2 + vy_mps**2)
    speed_knots = speed_mps * MPS_TO_KNOTS

    speed_px_s = displacement_px * FPS
    direction_rad = np.arctan2(-dy, dx) if dx or dy else 0
    direction_deg = np.degrees(direction_rad) % 360
    direction_norm = direction_deg / 360.0
    distance_to_origin_m = np.linalg.norm([x, y]) * PIXEL_TO_METER
    displacement_m = displacement_px * PIXEL_TO_METER
    tti_val = calculate_tti(x, y, dx, dy)
    tti_score = risk_from_tti(tti_val)
    custom_score = 0.5

    window = np.array([[x, y]*5]).reshape(1, -1)
    dx_pred, dy_pred = model5.predict(window)[0]
    reg_score = regression_risk_score(dx, dy, dx_pred, dy_pred)
    final_risk = 0.5 * custom_score + 0.3 * tti_score + 0.2 * reg_score

    speed_times_direction = speed_px_s * direction_deg
    tti_times_speed = tti_score * speed_px_s
    custom_times_tti = custom_score * tti_score

    return [
        custom_score, tti_score, reg_score,        # 0-2
        x, y, dx, dy,                              # 3-6
        displacement_px, displacement_m,           # 7-8
        speed_px_s, direction_deg,                 # 9-10
        distance_to_origin_m,                      # 11
        0, 0,                                      # 12-13 (placeholder)
        speed_times_direction, tti_times_speed,    # 14-15
        custom_times_tti,                          # 16
        final_risk,                                # 17
        x, y,                                      # 18-19
        vx_mps, vy_mps,                            # 20-21
        speed_mps, speed_knots                     # 22-23
    ]


def predict_transformer_risk(seq):
    if len(seq) < SEQ_LEN: return None
    input_seq = torch.tensor([list(seq)[-SEQ_LEN:]], dtype=torch.float32).to(device)
    with torch.no_grad():
        return model_time_series_transformer(input_seq).item()

def nash_adjusted_action(risk_score, base_probs=[0.2, 0.6, 0.2]):
    engage, evade, ignore = base_probs
    engage_adj = engage + 0.4 * risk_score
    ignore_adj = ignore + 0.4 * (1 - risk_score)
    evade_adj = 1.0 - (engage_adj + ignore_adj)
    probs = np.array([engage_adj, max(evade_adj, 0), ignore_adj])
    probs = np.maximum(probs, 0)
    probs /= probs.sum()
    actions = ["attack", "flee", "ignore"]
    choice = np.random.choice(actions, p=probs)
    return choice, dict(zip(actions, np.round(probs, 3)))


# ===== GPS Helpers =====
def destination_point(lat1_deg, lon1_deg, distance_m, bearing_deg):
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    bearing = math.radians(bearing_deg)

    d_km = distance_m / 1000.0
    lat2 = math.asin(math.sin(lat1) * math.cos(d_km / EARTH_RADIUS_KM) +
                     math.cos(lat1) * math.sin(d_km / EARTH_RADIUS_KM) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d_km / EARTH_RADIUS_KM) * math.cos(lat1),
                             math.cos(d_km / EARTH_RADIUS_KM) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)

def compute_bearing_from_image(cx, frame_width, cam_heading_deg=90, horizontal_fov_deg=60):
    image_center_x = frame_width / 2
    pixel_offset = cx - image_center_x
    bearing_offset = (pixel_offset / image_center_x) * (horizontal_fov_deg / 2)
    return (cam_heading_deg + bearing_offset) % 360


# ===== Image Processing Service =====
class ImageProcessingService:
    def __init__(self):
        self.cam_location_cache = {}  # {cam_id: (lat, lon, last_update_frame)}
        self.cam_frame_counter = {}   # {cam_id: frame_count}
        self.cam_update_interval = 30 # 30 frame’de bir güncelle

    def get_camera_location(self, cam_id):
        current_frame = self.cam_frame_counter.get(cam_id, 0)
        if cam_id in self.cam_location_cache:
            lat, lon, last_frame = self.cam_location_cache[cam_id]
            if current_frame - last_frame < self.cam_update_interval:
                return lat, lon

        url = f"http://localhost:8080/api/camera/{cam_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        lat, lon = data["latitude"], data["longitude"]

        self.cam_location_cache[cam_id] = (lat, lon, current_frame)
        return lat, lon

    def process_frame(self, frame, cam_id):
        self.cam_frame_counter[cam_id] = self.cam_frame_counter.get(cam_id, 0) + 1
        results = yolo_model.track(
            frame,
            conf=0.3,
            iou=0.5,
            tracker="services\\botsort.yaml",
            persist=True
        )

        obj_infos_dict = {}

        if results and results[0].boxes.id is not None:
            for det in results[0].boxes:
                obj_id = int(det.id.item())
                class_id = int(det.cls.item())
                class_name = yolo_model.names[class_id]
                conf = float(det.conf.item()) if det.conf is not None else 0.0

                features = compute_features(det, obj_id, buffer, regression_model)
                risk_score = features[17]

                buffer[obj_id].append(features[:17])
                pred_risk = predict_transformer_risk(buffer[obj_id]) or risk_score

                vx_mps, vy_mps, speed_mps, speed_knots = features[20], features[21], features[22], features[23]
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

                # Bearing hesapla
                cx = features[3]
                bearing_deg = compute_bearing_from_image(cx, frame.shape[1], cam_heading_deg=250, horizontal_fov_deg=60)

                # Kamera konumunu al
                cam_lat, cam_lon = self.get_camera_location(cam_id)
                distance_m = features[11]
                obj_lat, obj_lon = destination_point(cam_lat, cam_lon, distance_m, bearing_deg)

                # API’ye gönder
                payload = {
                    "obj_id": obj_id,
                    "latitude": obj_lat,
                    "longitude": obj_lon,
                    "last_seen": datetime.utcnow().isoformat() + "Z"
                }
                try:
                    requests.post("http://localhost:8080/api/location/insert", json=payload, timeout=1)
                except requests.RequestException as e:
                    print(f"Konum gönderilemedi: {e}")

                # Görsel çizimler
                color = (
                    (0, 0, 255) if pred_risk is None else
                    (255, 0, 0) if pred_risk > 75 else
                    (255, 255, 0) if pred_risk > 50 else
                    (0, 255, 0)
                )
                action, prob_dist = nash_adjusted_action(pred_risk / 100.0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{obj_id} | CONF:{conf:.2f} | RISK:{pred_risk:.1f} | CLASS:{class_name}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"V: {speed_mps:.2f} m/s | {round(float(features[11]), 3)} m",
                            (x1, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, action.upper(), (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

                # obj_infos’a ekle
                obj_infos_dict[obj_id] = {
                    "obj_id": obj_id,
                    "cam_id": cam_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "risk_score": round(pred_risk, 2),
                    "velocity_mps": {
                        "vx": round(vx_mps, 3),
                        "vy": round(vy_mps, 3),
                        "speed": round(speed_mps, 3),
                    },
                    "velocity_knots": round(speed_knots, 3),
                    "distance": round(float(features[11]), 3),
                    "bearing_deg": round(bearing_deg, 2),
                    "gps": {
                        "lat": obj_lat,
                        "lon": obj_lon
                    }
                }

        obj_infos = list(obj_infos_dict.values())
        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        return image_bytes, obj_infos



    """
    alarm_df = pd.DataFrame(risk_alarm_log)
    alarm_log_path = "./services/aimodels/risk_alarm_log.csv"
    alarm_df.to_csv(alarm_log_path, index=False)

    total_alarms = len(risk_alarm_log)
    efficiency = total_alarms / total_frames if total_frames else 0

    print(f"\n Video saved to: {OUTPUT_PATH}")
    if not alarm_df.empty:
        print("\n Alarms by Object ID:")
        print(alarm_df.groupby("object_id").size())
    """

            

"""
 def process_frame(self, frame, cam_id):
        # results = model(image_path)   # 2 ways to give image input
        result = model(frame)[0]

        obj_infos = []
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            risk_level = random.randint(0, 100)
            obj_infos.append({'obj_id': i, 'class_name': class_name, 'cam_id': cam_id, 'velocity': 100, 'risk_level': risk_level})

        # class_name = model.names[cls_id]
        # print("results is: ", results)
        rendered_image = result.plot()  # This returns a NumPy array with boxes and labels drawn
        encoded_image = cv2.imencode('.jpg', rendered_image)[1]
        image_bytes = encoded_image.tobytes()

        return image_bytes, obj_infos
 """   
    

"""
from ultralytics import YOLO
import cv2
import random

model = YOLO("./yolo11.pt")

class ImageProcessingService:
    def __init__(self):
        pass
    
    def process_frame(self, frame, cam_id):
        # results = model(image_path)   # 2 ways to give image input
        result = model(frame)[0]

        obj_infos = []
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            risk_level = random.randint(0, 100)
            obj_infos.append({'obj_id': i, 'class_name': class_name, 'cam_id': cam_id, 'velocity': 100, 'risk_level': risk_level})

        # class_name = model.names[cls_id]
        # print("results is: ", results)
        rendered_image = result.plot()  # This returns a NumPy array with boxes and labels drawn
        encoded_image = cv2.imencode('.jpg', rendered_image)[1]
        image_bytes = encoded_image.tobytes()

        return image_bytes, obj_infos
    
    def process_frame(self, frame):
        encoded_image = cv2.imencode('.jpg', frame)[1]
        
        image_bytes = encoded_image.tobytes()

        return image_bytes, {}
    
""" 