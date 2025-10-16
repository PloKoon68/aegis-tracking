import axios from "axios";

//const API_BASE_URL = "http://localhost:5000"; // Change this to your backend URL

const HTTP_URL =  'http' + process.env.REACT_APP_YOLO_SERVICE_URL;
console.log(HTTP_URL, "video/5")
console.log(HTTP_URL)
const axiosInstance = axios.create({
  baseURL: HTTP_URL,
  timeout: 60000, // 1 seconds timeout
  headers: {
    "Content-Type": "application/json",
  },
});

export default axiosInstance;
 
