import os
import math
import json
from time import timezone
import joblib
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import defaultdict, deque
import math
import requests
from datetime import datetime

# ========== YOLOX dedektörü ==========
# Kendi projenizdeki yola göre import edin
# detection_yolox/ klasörünüzde, verdiğiniz YOLO sınıfı olmalı.
from detection_yolox.yolo import YOLO as YOLOX
from deep_sort.deep_sort.deep_sort import DeepSort
from IPython import embed



# =====================================
#               CİHAZ
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================================
#             SABİTLER
# =====================================
SEQ_LEN = 30
RISK_THRESHOLD = 75
PIXEL_TO_METER = 0.2
MPS_TO_KNOTS = 1.94384
FPS = 30
EARTH_RADIUS_KM = 6371.0

# Model yolları
YOLOX_PATH = "detection_yolox/model_data/YOLOX-final.pth"
CLASSES_PATH = "detection_yolox\\model_data\\ship_classes.txt"
REGRESSOR_PATH = "./services/aimodels/model5.pkl"
TRANSFORMER_PATH = "./services/aimodels/transformer_model_weights.pth"


# =====================================
#         TRANSFORMER MODELİ
# =====================================
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



# =====================================
#           MODELLERİ YÜKLE
# =====================================
# YOLOX dedektörü
yolo_model = YOLOX(
    model_path=YOLOX_PATH,
    classes_path=CLASSES_PATH,
    confidence=0.3,  # istersen değiştir
    nms_iou=0.5
)

# # Basit tracker
# tracker = DeepSort(
#     model_path="deep_sort/deep/checkpoint/ckpt.t7",
#     max_dist=0.2,
#     min_confidence=0.3,
#     nms_max_overlap=1.0,
#     max_iou_distance=0.7,
#     max_age=30,
#     n_init=3,
#     nn_budget=100
# )

# Regresyon modeli
regression_model = joblib.load(REGRESSOR_PATH)

# Zaman serisi risk modeli
model_time_series_transformer = TimeSeriesTransformer(17, 128, 4, 4).to(device)
model_time_series_transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device))
model_time_series_transformer.eval()


# =====================================
#        DURUM & BUFFER YAPILARI
# =====================================
buffer = defaultdict(lambda: deque(maxlen=SEQ_LEN))
risk_alarm_log = []


# =====================================
#          YARDIMCI FONKSİYONLAR
# =====================================
def xyxy_to_xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h

def box_iou_xyxy(a, b):
    # a: [x1,y1,x2,y2], b: [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def match_track_to_det(track_box, det_boxes_xyxy, iou_thresh=0.1):
    # track_box: [x1,y1,x2,y2]
    # det_boxes_xyxy: list of [x1,y1,x2,y2]
    best_iou, best_idx = 0.0, -1
    for i, det_box in enumerate(det_boxes_xyxy):
        iou = box_iou_xyxy(track_box, det_box)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return (best_idx if best_iou >= iou_thresh else -1)


def extract_displacement(dx, dy):
    return math.sqrt(dx**2 + dy**2)

def calculate_tti(x, y, dx, dy):
    pos = np.array([x, y])
    vel = np.array([dx, dy])
    distance = np.linalg.norm(pos)
    if distance == 0:
        return 0
    rel_speed = np.dot(pos, vel) / distance
    return float("inf") if rel_speed <= 0 else distance / rel_speed

def risk_from_tti(tti):
    if tti == float("inf"):
        return 0
    elif tti < 3:
        return 25
    elif tti < 6:
        return 15
    elif tti < 10:
        return 8
    else:
        return 2

def regression_risk_score(dx_true, dy_true, dx_pred, dy_pred):
    err = np.sqrt((dx_true - dx_pred)**2 + (dy_true - dy_pred)**2)
    return min(err * 10, 100)

def compute_features(cx, cy, w, h, object_id, buffer, model5):
    frame_data = buffer.get(object_id, [])
    dx = dy = 0.0

    if len(frame_data) > 0:
        prev_x, prev_y = frame_data[-1][3], frame_data[-1][4]
        dx = cx - prev_x
        dy = cy - prev_y

    displacement_px = extract_displacement(dx, dy)

    vx_mps = dx * PIXEL_TO_METER * FPS
    vy_mps = dy * PIXEL_TO_METER * FPS
    speed_mps = math.sqrt(vx_mps**2 + vy_mps**2)
    speed_knots = speed_mps * MPS_TO_KNOTS

    speed_px_s = displacement_px * FPS
    direction_rad = np.arctan2(-dy, dx) if (dx != 0 or dy != 0) else 0.0
    direction_deg = float(np.degrees(direction_rad) % 360)
    distance_to_origin_m = float(np.linalg.norm([cx, cy]) * PIXEL_TO_METER)
    displacement_m = displacement_px * PIXEL_TO_METER
    tti_val = calculate_tti(cx, cy, dx, dy)
    tti_score = risk_from_tti(tti_val)
    custom_score = 0.5

    window = np.array([[cx, cy] * 5]).reshape(1, -1)
    dx_pred, dy_pred = model5.predict(window)[0]
    reg_score = regression_risk_score(dx, dy, dx_pred, dy_pred)
    final_risk = 0.5 * custom_score + 0.3 * tti_score + 0.2 * reg_score

    speed_times_direction = speed_px_s * direction_deg
    tti_times_speed = tti_score * speed_px_s
    custom_times_tti = custom_score * tti_score

    return [
        custom_score, tti_score, reg_score,
        cx, cy, dx, dy,
        displacement_px, displacement_m,
        speed_px_s, direction_deg,
        distance_to_origin_m,
        0, 0,
        speed_times_direction,
        tti_times_speed,
        custom_times_tti,
        final_risk,
        cx, cy,
        vx_mps, vy_mps,
        speed_mps, speed_knots
    ]


def compute_bearing_from_image(cx, frame_width, cam_heading_deg=90, horizontal_fov_deg=60):
    image_center_x = frame_width / 2
    pixel_offset = cx - image_center_x
    bearing_offset = (pixel_offset / image_center_x) * (horizontal_fov_deg / 2)
    return (cam_heading_deg + bearing_offset) % 360


def predict_transformer_risk(seq):
    if len(seq) < SEQ_LEN:
        return None
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

def destination_point(lat1_deg, lon1_deg, distance_m, bearing_deg):
    """
    lat1_deg, lon1_deg: başlangıç noktası (derece)
    distance_m: metre cinsinden mesafe
    bearing_deg: kuzeyden saat yönünde derece
    """
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    bearing = math.radians(bearing_deg)

    d_km = distance_m / 1000.0
    lat2 = math.asin(math.sin(lat1) * math.cos(d_km / EARTH_RADIUS_KM) +
                     math.cos(lat1) * math.sin(d_km / EARTH_RADIUS_KM) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d_km / EARTH_RADIUS_KM) * math.cos(lat1),
                             math.cos(d_km / EARTH_RADIUS_KM) - math.sin(lat1) * math.sin(lat2))

    return math.degrees(lat2), math.degrees(lon2)



class ImageProcessingService:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
        # BASE_DIR = mvc/backend/ai

        CKPT_PATH = os.path.join(
            BASE_DIR,
            "deep_sort", "deep_sort", "deep", "checkpoint", "ckpt.t7"
        )
        self.deepsort = DeepSort(
            model_path=CKPT_PATH,
            max_dist=0.2,
            min_confidence=0.3,
            nms_max_overlap=1.0,
            max_iou_distance=0.7,
            max_age=30,
            n_init=3,
            nn_budget=100
        )
        self.cam_location_cache = {}  # {cam_id: (lat, lon, last_update_frame)}
        self.cam_frame_counter = {}  # {cam_id: frame_count}
        self.cam_update_interval = 30  # 30 frame’de bir güncelle

    def get_camera_location(self, cam_id):
        current_frame = self.cam_frame_counter.get(cam_id, 0)

        if cam_id in self.cam_location_cache:
            lat, lon, last_frame = self.cam_location_cache[cam_id]
            if current_frame - last_frame < self.cam_update_interval:
                return lat, lon

        # API'den çek
        url = f"http://localhost:8080/api/camera/{cam_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        lat, lon = data["latitude"], data["longitude"]

        # Cache’e yaz
        self.cam_location_cache[cam_id] = (lat, lon, current_frame)
        return lat, lon


    def process_frame(self, frame, cam_id):
        self.cam_frame_counter[cam_id] = self.cam_frame_counter.get(cam_id, 0) + 1
        detections_raw = yolo_model.detect_image(frame)

        det_boxes_xyxy, det_classes, det_confs = [], [], []
        bbox_xywh_list, conf_list = [], []

        for (x1, y1, x2, y2, cls_name, score_tensor) in detections_raw:
            conf = float(score_tensor.detach().cpu().numpy()) if isinstance(score_tensor, torch.Tensor) else float(score_tensor)
            cls_id = yolo_model.class_names.index(cls_name)
            det_boxes_xyxy.append([x1, y1, x2, y2])
            det_classes.append(cls_id)
            det_confs.append(conf)
            cx, cy, w, h = xyxy_to_xywh(x1, y1, x2, y2)
            bbox_xywh_list.append([cx, cy, w, h])
            conf_list.append(conf)

        bbox_xywh_tensor = torch.tensor(bbox_xywh_list, dtype=torch.float32) if bbox_xywh_list else torch.empty((0, 4))
        confidences_tensor = torch.tensor(conf_list, dtype=torch.float32) if conf_list else torch.empty((0,))

        outputs = self.deepsort.update(
            bbox_xywh=bbox_xywh_tensor,
            confidences=confidences_tensor,
            ori_img=frame,
            bbox_xywh_anti_occ=torch.empty((0, 4)),
            confidences_anti_occ=torch.empty((0,))
        )

        obj_infos_dict = {}

        for x1, y1, x2, y2, lines, track_id in outputs:
            best_idx = match_track_to_det([x1, y1, x2, y2], det_boxes_xyxy, iou_thresh=0.1)
            cls_id = det_classes[best_idx] if best_idx >= 0 else -1
            conf = det_confs[best_idx] if best_idx >= 0 else 0.0
            class_name = yolo_model.class_names[cls_id] if cls_id >= 0 else "unknown"

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            object_id = cam_id * 1000 + track_id

            if object_id not in buffer:
                buffer[object_id] = []

            features = compute_features(cx, cy, w, h, object_id, buffer, regression_model)
            bearing_deg = compute_bearing_from_image(cx, frame.shape[1], cam_heading_deg=250, horizontal_fov_deg=60)
            features[10] = bearing_deg

            risk_score = features[17]
            buffer[object_id].append(features[:17])
            pred_risk = predict_transformer_risk(buffer[object_id]) or risk_score

            vx_mps, vy_mps, speed_mps, speed_knots = features[20], features[21], features[22], features[23]

            color = (0, 0, 255) if pred_risk is None else (255, 0, 0) if pred_risk > 75 else (255, 255, 0) if pred_risk > 50 else (0, 255, 0)
            action, prob_dist = nash_adjusted_action((pred_risk or 0) / 100.0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID:{track_id} | CONF:{conf:.2f} | RISK:{pred_risk:.1f} | CLASS:{class_name}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"V: {speed_mps:.2f} m/s | {round(features[11], 3)} m", (int(x1), int(y1) + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, action.upper(), (int(x1) + 5, int(y1) + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

            if pred_risk > RISK_THRESHOLD:
                risk_alarm_log.append({
                    "obj_id": object_id,
                    "risk_score": round(pred_risk, 2),
                    "action": action
                })

            cam_lat, cam_lon = self.get_camera_location(cam_id)
            distance_m = features[11]
            bearing_deg = features[10]
            obj_lat, obj_lon = destination_point(cam_lat, cam_lon, distance_m, bearing_deg)

            payload = {
                "obj_id": object_id,
                "latitude": obj_lat,
                "longitude": obj_lon,
                "last_seen": datetime.utcnow().isoformat() + "Z"
            }
            try:
                requests.post("http://localhost:8080/api/location/insert", json=payload, timeout=1)
            except requests.RequestException as e:
                print(f"Konum gönderilemedi: {e}")

            obj_infos_dict[object_id] = {
                "obj_id": object_id,
                "cam_id": cam_id,
                "class_name": class_name,
                "confidence": float(conf),
                "risk_score": round(pred_risk, 2),
                "velocity_mps": {
                    "vx": round(vx_mps, 3),
                    "vy": round(vy_mps, 3),
                    "speed": round(speed_mps, 3),
                },
                "velocity_knots": round(speed_knots, 3),
                "distance": round(distance_m, 3),
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