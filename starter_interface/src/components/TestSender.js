import React, { useState, useRef } from "react";
import { Card, CardContent, Typography, TextField, Button, Stack } from "@mui/material";
import axios from "axios";

export default function TestSender() {
  const [testObjId, setTestObjId] = useState("");
  const [speedLatInput, setSpeedLatInput] = useState("");
  const [speedLongInput, setSpeedLongInput] = useState("");
  const intervalRef = useRef(null);

  const startTestSending = async () => {
    if (!testObjId || !speedLatInput || !speedLongInput) {
      return alert("Obj ID, Speed Latitude ve Speed Longitude giriniz!");
    }

    try {
      // API'den başlangıç konumunu çek
      const res = await axios.get(`/api/location/${testObjId}`)

      if (!res.data || res.data.length === 0) {
        return alert("Belirtilen Obj ID için konum bulunamadı!");
      }
      console.log(res.data[0]);
      let lat = Number(res.data[0].latitude);
      let lon = Number(res.data[0].longitude);
      

      // Kullanıcının girdiği hız değerleri
      let speedLat = Number(speedLatInput);
      let speedLong = Number(speedLongInput);

      console.log(`Başlangıç konumu: lat=${lat}, lon=${lon}`);
      console.log(`Hızlar: speedLat=${speedLat}, speedLong=${speedLong}`);

      intervalRef.current = setInterval(async () => {
        const payload = {
          obj_id: Number(testObjId),
          latitude: lat,
          longitude: lon,
          last_seen: new Date().toISOString()
        };

        try {
          await axios.post("http://localhost:8080/api/location/insert", payload);
          console.log("Gönderildi:", payload);
        } catch (err) {
          console.error("Hata oluştu:", err);
        }

        // Koordinatları hızlara göre güncelle
        lat += speedLat;
        lon += speedLong;
      }, 500);

    } catch (err) {
      console.error("Başlangıç konumu alınırken hata:", err);
      alert("Başlangıç konumu alınamadı!");
    }
  };

  const stopTestSending = () => {
    clearInterval(intervalRef.current);
    intervalRef.current = null;
    console.log("Test gönderimi durduruldu.");
  };

  return (
    <Card sx={{ boxShadow: 3 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          🧪 Test Konum Gönderimi
        </Typography>
        <Stack spacing={2}>
          <TextField
            label="Test Obj ID"
            type="number"
            value={testObjId}
            onChange={(e) => setTestObjId(e.target.value)}
            fullWidth
          />
          <TextField
            label="Speed Latitude"
            type="number"
            value={speedLatInput}
            onChange={(e) => setSpeedLatInput(e.target.value)}
            fullWidth
          />
          <TextField
            label="Speed Longitude"
            type="number"
            value={speedLongInput}
            onChange={(e) => setSpeedLongInput(e.target.value)}
            fullWidth
          />
          <Stack direction="row" spacing={2}>
            <Button
              variant="contained"
              color="success"
              onClick={startTestSending}
              fullWidth
            >
              Testi Başlat
            </Button>
            <Button
              variant="contained"
              color="error"
              onClick={stopTestSending}
              fullWidth
            >
              Testi Durdur
            </Button>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
