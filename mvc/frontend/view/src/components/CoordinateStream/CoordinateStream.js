import { useEffect, useState } from 'react';

import './CoordinateStream.css';


function CoordinateStream() {
  const [coordinate, setCoordinate] = useState("");

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:5000/ws');

    ws.onopen = () => {
      console.log("Connected to Flask WebSocket");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received:", data);
      setCoordinate(data);
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="log" style={{padding:"50px"}}>
      <h2 style={{marginTop:"0px"}}>Live Targets</h2>
      ID: {coordinate.id}, X: {coordinate.x}, Y: {coordinate.y}, Z: {coordinate.z}
    </div>
  );
}

export default CoordinateStream;
