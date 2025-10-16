import { useEffect, useState } from 'react';

import './Bounding Boxes.css';

//change test
function BoundingBox() {

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:5000/ws-video");
    socket.binaryType = "blob";

    socket.onmessage = function(event) {
        const blob = event.data;
        const url = URL.createObjectURL(blob);
        document.getElementById("image").src = url;
    };

     socket.onclose = () => {
      console.log("WebSocket disconnected");
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      socket.close();
    };
  }, []);
   

  return (
    <div className="image-container" style={{padding:"50px"}}>
      <img id="image" />
    </div>
  );
}

export default BoundingBox;
