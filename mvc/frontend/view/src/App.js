import './App.css';

import { Route, Routes, Navigate, useLocation } from 'react-router-dom'; 


import CoordinateStream from "./components/CoordinateStream/CoordinateStream"
import Navbar from './components/Navbar/Navbar';
import BoundingBox from './components/Bounding Boxes/Bounding Boxes';
import Map from './components/Map/Map';

function App() {
  return (
    <div className="App">        
      <Navbar/>
      <Routes>
        <Route path="/" element={<Navigate to="/map" replace />} />
        <Route path="/map" element={<Map />} />
        {/* <Route path="/logs" element={<CoordinateStream />} />
        <Route path="/bounding-box" element={<BoundingBox />} /> */}
      </Routes>
    </div>
  );
}

export default App;

/*
    <Routes>
        <Route path="/bounding-box" element={<Navigate to="/bounding-box" />} />
        <Route path="/map" element={<BoundingBox/>} />
      </Routes>
*/