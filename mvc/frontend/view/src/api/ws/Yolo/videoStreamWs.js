const YOLO_SERVICE_VIDEO_STREAM_WS_URL =  'ws' + process.env.REACT_APP_YOLO_SERVICE_URL + 'ws/ai-video-stream';

export const generateVideoStreamSocket = async () => {

    const ws = new WebSocket(YOLO_SERVICE_VIDEO_STREAM_WS_URL); // Use your Crow port, often 18080
    ws.binaryType = "blob";
    console.log("yolo url: ", YOLO_SERVICE_VIDEO_STREAM_WS_URL);
    ws.onopen = () => {
        console.log("WebSocket for video stream connected");
    };


    ws.onmessage = function(event) {
        const blob = event.data;
        const url = URL.createObjectURL(blob);
        document.getElementById("image").src = url;
    };

     ws.onclose = () => {
      console.log("WebSocket for video stream disconnected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error for video streaming:", error);
    };


    return ws
}

