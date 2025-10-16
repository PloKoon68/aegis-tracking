import requests
import time
from datetime import datetime

url = "http://127.0.0.1:8080/api/location/insert"

# Sabit obj_id ve koordinatlar
obj_id = 1000
latitude = 41.39
longitude = 29.22
speed_lat = 0.09
speed_long = 0.09
a = 0
state = 1

while True:
    # Şu anki zamanı ISO formatında al
    current_time = datetime.utcnow().isoformat() + "Z"

    # Gönderilecek veri
    data = {
        "obj_id": obj_id,
        "latitude": latitude,
        "longitude": longitude,
        "last_seen": current_time
    }

    try:
        response = requests.post(url, json=data)
        print(f"Gönderildi: {data} | Durum: {response.status_code}")
        latitude = latitude +speed_lat
        longitude = longitude +speed_long
        a = a +1
        if a == 5:
            latitude = latitude +0.2
        if latitude > 43.5:
            speed_lat = 0
            speed_long = 0.09
        if longitude > 37.0:
            speed_lat = -0.09
            speed_long = 0.01
        if latitude < 42.5:
            speed_lat = 0.00
            speed_long = -0.09
        if longitude < 29.0:
            speed_lat = 0.09
            speed_long = 0.09
    except Exception as e:
        print(f"Hata oluştu: {e}")

    time.sleep(0.5)
