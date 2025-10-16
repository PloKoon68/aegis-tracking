import React, { useState } from "react";
import { Card, CardContent, Typography, TextField, Button, Stack } from "@mui/material";
import axios from "axios";

export default function LocationForm() {
  const [objId, setObjId] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");

  const handleLocationInsert = async () => {
    if (!objId || !latitude || !longitude)
      return alert("Lütfen tüm konum bilgilerini girin!");

    const payload = {
      obj_id: Number(objId),
      latitude: Number(latitude),
      longitude: Number(longitude),
      last_seen: new Date().toISOString()
    };

    try {
      await axios.post("http://localhost:8080/api/location/insert", payload);
      alert("Konum başarıyla gönderildi!");
    } catch {
      // alert("Konum gönderilirken hata oluştu!");
    }
  };

  return (
    <Card sx={{ mb: 4, boxShadow: 3 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          📍 Konum Gönder
        </Typography>
        <Stack spacing={2}>
          <TextField
            label="Obj ID"
            type="number"
            value={objId}
            onChange={(e) => setObjId(e.target.value)}
            fullWidth
          />
          <TextField
            label="Latitude"
            type="number"
            value={latitude}
            onChange={(e) => setLatitude(e.target.value)}
            fullWidth
          />
          <TextField
            label="Longitude"
            type="number"
            value={longitude}
            onChange={(e) => setLongitude(e.target.value)}
            fullWidth
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleLocationInsert}
            fullWidth
          >
            Konum Gönder
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
}
