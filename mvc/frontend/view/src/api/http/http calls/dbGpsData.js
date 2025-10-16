import axios from "axios";
const HTTP_URL =  'http' + process.env.REACT_APP_GPS_SERVICE_URL + 'api/location';

const fetchGpsData = async (activeTrackIdRef, setDetectedObjects) => {
  try {
    console.log("waiting for coordinates");
    const coordinates = (await axios.get(HTTP_URL)).data;

    setDetectedObjects(prev => {
      const updated = { ...prev };
      const activeId = String(activeTrackIdRef?.current || "");

      coordinates.forEach(obj => {
        const { obj_id, latitude, longitude, altitude, last_seen } = obj;

        // Geçersiz obj_id veya aktif takip edilen objeyi atla
        if (!obj_id || (activeId && obj_id === activeId)) {
          return;
        }

        // Diğer objeleri güncelle
        updated[obj_id] = {
          ...updated[obj_id],
          gps_data: {
            latitude,
            longitude,
            altitude,
            last_seen,
            recievedState: "not received"
          },
          showInfo: false
        };
      });

      // Aktif objeyi silme, sadece olduğu gibi bırak
      console.log("Updated objects:", updated);
      return updated;
    });

  } catch (err) {
    console.error("GPS data fetch error:", err);
    return err;
  }
};





const fetchGpsDataByID = async ( id) => {
  try {
    console.log("Fetching GPS coordinates...");
    const coordinates = (await axios.get(`${HTTP_URL}/${id}`)).data;

    // Doğrudan dizi olarak set et
    console.log("data alındı",id, coordinates);

    return coordinates;
  } catch (err) {
    console.error("GPS data fetch failed:", err);
    return null;
  }
};




export {
  fetchGpsData, fetchGpsDataByID
};
