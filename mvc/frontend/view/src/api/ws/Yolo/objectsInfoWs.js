const YOLO_SERVICE_OBJECT_INFO_WS_URL =  'ws' + process.env.REACT_APP_YOLO_SERVICE_URL + 'ws/object-info';

export const generateObjectInfoWs = async (setDetectedObjects, dete) => {

    const ws = new WebSocket(YOLO_SERVICE_OBJECT_INFO_WS_URL); // Use your Crow port, often 18080

    ws.onopen = () => {
        console.log("WebSocket for object info connected");
    };


    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDetectedObjects(prev => {
      const updated = { ...prev };

      const incomingIds = data.map(obj => obj.obj_id);

      Object.keys(updated).forEach(obj_id => {
        if (!incomingIds.includes(obj_id)) {
          updated[obj_id] = {
            ...updated[obj_id],
            isFresh: false
          };
        }
      });

      data.forEach(obj => {
        const { obj_id, cam_id, class_name, confidence, velocity_mps, risk_score } = obj;
        const newInfo = { cam_id, class_name, confidence, velocity_mps, risk_score };

        const isExisting = !!updated[obj_id];
        const oldInfo = updated[obj_id]?.obj_info;

        const isChanged = JSON.stringify(oldInfo) !== JSON.stringify(newInfo);

        const isFresh = !isExisting || isChanged;

        updated[obj_id] = {
          ...updated[obj_id],
          obj_info: newInfo,
          showInfo: false,
          isFresh: isFresh
        };
      });

      return updated;
    });


    };



     ws.onclose = () => {
      console.log("WebSocket for object info disconnected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error for object info:", error);
    };


    return ws
}

