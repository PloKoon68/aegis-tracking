const GPS_SERVİCE_GPS_DATA_WS_URL =  'ws' + process.env.REACT_APP_GPS_SERVICE_URL + 'ws/gps-data';

export const generateGpsDataWs = async (setDetectedObjects, activeTrackIdRef) => {

    const ws = new WebSocket(GPS_SERVİCE_GPS_DATA_WS_URL); 
    console.log("url", GPS_SERVİCE_GPS_DATA_WS_URL);

    ws.onopen = () => {
        console.log("WebSocket for gps data connected");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const { obj_id, latitude, longitude, last_seen } = data;

      // Eğer activeTrackIdRef tanımlı ve bu objeye ait id eşleşiyorsa atla
      if (activeTrackIdRef?.current && obj_id === String(activeTrackIdRef.current)) {
        return;
      }
      setDetectedObjects(prev => {
        const updated = { ...prev };
        const activeId = String(activeTrackIdRef?.current);

        // Eğer aktif takip edilen obje varsa ve gelen veri ona aitse, hiçbir şey yapma
        if (activeId && obj_id === activeId) {
          return updated;
        }

        // Diğer objeleri güncelle
        updated[obj_id] = {
          ...updated[obj_id],
          gps_data: {
            latitude,
            longitude,
            last_seen,
            recievedState: "received"
          },
          showInfo: false
        };

        return updated;
      });


    };

    ws.onclose = () => {
      console.log("WebSocket for gps data disconnected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error for gps data:", error);
    };

    return ws;
};
