import axios from "axios";

const CAMERA_API_URL = "http://localhost:8080/api/camera";

const fetchCameraData = async (setCameraObjects) => {
  try {
    console.log("Fetching camera data...");
    const cameraData = (await axios.get(CAMERA_API_URL)).data;

    setCameraObjects(prev => {
      const updated = { ...prev };

      cameraData.forEach(obj => {
        const { id, camID, latitude, longitude, type, lastSeen } = obj;

        // Kamera objesini güncelle
        updated[id] = {
          ...updated[id],
          camera_data: {
            camID,
            latitude,
            longitude,
            type,
            lastSeen,
            recievedState: "not received"
          },
          showInfo: false
        };
      });

      console.log("Updated camera objects:", updated);
      return updated;
    });

  } catch (error) {
    console.error("Camera data fetch error:", error);
    return error;
  }
};

export{
    fetchCameraData
}
