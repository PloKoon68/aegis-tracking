import axiosInstance from "../axios"; // Import the axios instance

let lastCamId = null;

const requestVideo = async (cam_id) => {
  try {
    console.log("sending: ", cam_id);
    lastCamId = cam_id;
    await axiosInstance.post(`video/${cam_id}`);
  } catch (err) {
    return err;
  }
};

const getLastCamId = () => lastCamId;

export {
  requestVideo,
  getLastCamId
};

