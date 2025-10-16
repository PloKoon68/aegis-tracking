import React, { useState } from "react";
import { Card, CardContent, Typography, TextField, Button, Stack } from "@mui/material";
import axios from "axios";

export default function VideoControl() {
  const [videoName, setVideoName] = useState("");

  const handleVideoRequest = async (action) => {
    if (!videoName) return alert("Lütfen video adını girin!");
    const url = `http://localhost:8091/api/streams/${action}/video${videoName}`;
    try {
      await axios.post(url);
      alert(`${action === "start" ? "Başlatma" : "Durdurma"} isteği gönderildi!`);
    } catch {
      // alert("İstek gönderilirken hata oluştu!");
    }
  };

  return (
    <Card sx={{ mb: 4, boxShadow: 3 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          🎥 Video Stream Kontrol
        </Typography>
        <Stack spacing={2}>
          <TextField
            label="Video Adı"
            variant="outlined"
            value={videoName}
            onChange={(e) => setVideoName(e.target.value)}
            fullWidth
          />
          <Stack direction="row" spacing={2}>
            <Button
              variant="contained"
              color="success"
              onClick={() => handleVideoRequest("start")}
              fullWidth
            >
              Başlat
            </Button>
            <Button
              variant="contained"
              color="error"
              onClick={() => handleVideoRequest("stop")}
              fullWidth
            >
              Durdur
            </Button>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
