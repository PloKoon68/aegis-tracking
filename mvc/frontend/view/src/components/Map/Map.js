import './Map.css';
import "leaflet/dist/leaflet.css";
import 'bootstrap/dist/css/bootstrap.css';
import { MapContainer, Marker, Popup, TileLayer, useMap, Polygon, Circle, Rectangle } from "react-leaflet";
import { Icon, DrawEvents, Polyline} from "leaflet";
import 'leaflet.offline';
import L from 'leaflet';
import "leaflet-draw/dist/leaflet.draw.css";
import "leaflet-draw";
import { useEffect, useState, useMemo, useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';


import {requestVideo, getLastCamId} from '../../api/http/http calls/videoRequest.js'
import {fetchGpsData, fetchGpsDataByID} from '../../api/http/http calls/dbGpsData.js'
import {fetchLocationsInsideShapes, checkAnyChanges} from '../../api/http/http calls/dbDangerInfo.js'
import {addFetchedShapesToMap, fetchShapeData} from '../../api/http/http calls/dbShapeData.js'
import {fetchCameraData} from '../../api/http/http calls/dbCameraData.js'



import {generateVideoStreamSocket} from '../../api/ws/Yolo/videoStreamWs.js'
import {generateObjectInfoWs} from '../../api/ws/Yolo/objectsInfoWs.js'
import {generateGpsDataWs} from '../../api/ws/Gps/gpsDataWs.js'


function MapInitializer({ setMap, onShapeCreated, drawnItemsRef, setDetectedShapes }) {
  const map = useMap();

  useEffect(() => {
    if (!map) return;

    setMap(map);

    map.addLayer(drawnItemsRef.current);

    const drawControl = new L.Control.Draw({
      edit: { featureGroup: drawnItemsRef.current },
      draw: {
        polygon: true,
        rectangle: true,
        polyline: false,
        circle: true,
        marker: false
      }
    });
    map.addControl(drawControl);

    const handleCreated = async (e) => {
      const layer = e.layer;
      // drawnItemsRef.current.addLayer(layer);

      const geojson = layer.toGeoJSON();
      console.log("Yeni şekil:", geojson);

      const geometry = geojson.geometry;

      if (!geometry || !geometry.coordinates) {
        console.error("Geometri bilgisi eksik veya geçersiz.");
        return;
      }

      try {
        if (geometry.type === "Polygon") {
          const coords = geometry.coordinates[0];

          if (!Array.isArray(coords) || coords.length < 3) {
            console.error("Polygon için yeterli köşe yok.");
            return;
          }

          console.log("Koordinatlar:", coords);

          if (coords.length === 5) {
            const rectangleDto = {
              c1: { latitude: coords[0][0], longitude: coords[0][1] },
              c2: { latitude: coords[1][0], longitude: coords[1][1] },
              c3: { latitude: coords[2][0], longitude: coords[2][1] },
              c4: { latitude: coords[3][0], longitude: coords[3][1] },
              name: "Dikdortgen"
            };

            console.log("Dikdörtgen DTO:", rectangleDto);

            const res = await fetch("http://127.0.0.1:8080/api/shape/rectangle/insert", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(rectangleDto)
            });
            const data = await res.json();
            console.log("Dikdörtgen yanıtı:", data);

          } else {
            const polygonDto = {
              points: coords.map(([lng, lat]) => ({ latitude: lat, longitude: lng })),
              name: "Yeni Polygon"
            };

            console.log("Polygon DTO:", polygonDto);

            const res = await fetch("http://127.0.0.1:8080/api/shape/polygon/insert", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(polygonDto)
            });
            const data = await res.json();
            console.log("Polygon yanıtı:", data);
          }

        } else if (geometry.type === "Point" && layer instanceof L.Circle) {
          const center = layer.getLatLng();
          const radius = layer.getRadius();

          const circleDto = {
            center: { latitude: center.lat, longitude: center.lng },
            radius,
            name: "Yeni Daire"
          };

          console.log("Daire DTO:", circleDto);

          const res = await fetch("http://127.0.0.1:8080/api/shape/circle/insert", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(circleDto)
          });
          const data = await res.json();
          console.log("Daire yanıtı:", data);
        } else {
          console.warn("Desteklenmeyen şekil türü:", geometry.type);
        }

        // ✅ DB'ye kayıt tamamlandıktan sonra şekilleri yeniden çek
        await fetchShapeData(setDetectedShapes, drawnItemsRef);

        if (onShapeCreated) {
          onShapeCreated(geojson);
        }

      } catch (err) {
        console.error("Şekil gönderim hatası:", err);
      }
    };

    const handleDeleted = (e) => {
      e.layers.eachLayer(async (layer) => {
        const geojson = layer.toGeoJSON();
        console.log("Silinen şekil:", geojson);

        const shapeId = layer._shapeId;
        if (!shapeId) {
          console.warn("Silinen şeklin ID'si bulunamadı, DB'den silinemiyor.");
          return;
        }

        try {
          const res = await fetch(`http://127.0.0.1:8080/api/shape/delete/${shapeId}`, {
            method: "DELETE"
          });

          if (!res.ok) {
            throw new Error(`Sunucu hatası: ${res.status}`);
          }

          const text = await res.text();
          if (text) {
            try {
              const data = JSON.parse(text);
              console.log("Silme yanıtı:", data);
            } catch (err) {
              console.warn("Yanıt JSON formatında değil:", text);
            }
          } else {
            console.log("Silme başarılı, sunucudan içerik dönmedi.");
          }

          // İstersen burada şekilleri yeniden çekebilirsin:
          // await fetchShapeData(setDetectedShapes, drawnItemsRef);

        } catch (err) {
          console.error("Silme hatası:", err);
        }
      });
    };



    map.on(L.Draw.Event.CREATED, handleCreated);
    map.on(L.Draw.Event.DELETED, handleDeleted);

    return () => {
      map.off(L.Draw.Event.CREATED, handleCreated);
      map.off(L.Draw.Event.DELETED, handleDeleted);
      map.removeControl(drawControl);
      map.removeLayer(drawnItemsRef.current);
    };
  }, [map, setMap, onShapeCreated]);
  return null;
}


function Map() {
  const [detectedObjects, setdetectedObjects] = useState({});
  const [detectedObjectLocations, setDetectedObjectLocations] = useState({});
  const [trackingFilter, setTrackingFilter] = useState("0");
  const [activeTrackId, setActiveTrackId] = useState(null);
  const [showInfoMap, setShowInfoMap] = useState({});
  const [dangerInfoList, setDangerInfoList] = useState([]);
  const [changedItems, setChangedItems] = useState([]);
  const [activePolyline, setActivePolyline] = useState([]);
  const [isAny, setIsAny] = useState(false);
  const [sliderIndex, setSliderIndex] = useState(0);
  const [cameraObjects, setCameraObjects] = useState({});


  let videoPort = 8552;
  const drawnItemsRef = useRef(new L.FeatureGroup());

  const sortedObjectsArray = useMemo(() => {
    return Object.entries(detectedObjects)
      .sort(([, a], [, b]) => {
        // Önce isFresh'e göre sırala (true olanlar önce)
        if (a.isFresh !== b.isFresh) {
          return b.isFresh - a.isFresh;
        }

        // Sonra risk_score'a göre sırala (yüksek olan önce)
        const aScore = a.obj_info?.risk_score ?? 0;
        const bScore = b.obj_info?.risk_score ?? 0;
        return bScore - aScore;
      })
      .map(([obj_id, data]) => ({
        obj_id,
        ...data
      }));
  }, [detectedObjects]);

  const filteredObjects = useMemo(() => {
    if (trackingFilter === "1") {
      const seenCamIds = new Set();
      return sortedObjectsArray.filter(obj => {
        const cam_id = obj.obj_info?.cam_id;
        if (!cam_id || seenCamIds.has(cam_id)) return false;
        seenCamIds.add(cam_id);
        return true;
      });
    } else if (trackingFilter === "2") {
      return sortedObjectsArray.filter(obj => obj.gps_data);
    }
    return sortedObjectsArray;
  }, [sortedObjectsArray, trackingFilter]);

  const polylinePositions = useMemo(
    () => activePolyline.filter(p => Array.isArray(p) && p.length === 2),
    [activePolyline]
  );

  const drawPolylineForObject = async (obj_id) => {
    const history = await fetchGpsDataByID(obj_id);
    console.log("history: ",history);
    setDetectedObjectLocations(history);
    setSliderIndex(history.length-1);
    console.log("detectedObject: ",detectedObjectLocations);

    if (Array.isArray(history) && history.length > 0) {
      const sortedHistory = [...history].sort(
        (a, b) => new Date(a.last_seen) - new Date(b.last_seen)
      );
      const path = sortedHistory.map(loc => [loc.latitude, loc.longitude]);

      // Eski çizgiyi kaldır
      if (polylineRef.current) {
        map.removeLayer(polylineRef.current);
      }
      //setSliderIndex(sortedHistory.length - 1);



      // 1
      // polylineRef.current = L.polyline(path, {
      //   color: '#4A90E2',        // Daha yumuşak mavi
      //   weight: 5,               // Kalınlık
      //   opacity: 0.8,            // Hafif şeffaf
      //   dashArray: '10, 10',     // Kesik çizgi efekti
      //   lineJoin: 'round',       // Köşeler yuvarlak
      //   lineCap: 'round'         // Uçlar yuvarlak
      // }).addTo(map);

      //2
      polylineRef.current = L.polyline(path, {
        color: '#00BFFF',
        weight: 5,
        opacity: 0.9,
        dashArray: '15, 10',
        className: 'animated-polyline'
      }).addTo(map);

      //3
      // polylineRef.current = L.polyline(path, {
      //   color: '#1E90FF',
      //   weight: 6,
      //   opacity: 0.85,
      //   dashArray: '12, 8',
      //   lineJoin: 'round',
      //   lineCap: 'round',
      //   className: 'shadowed-polyline'
      // }).addTo(map);

      // map.fitBounds(polylineRef.current.getBounds());
    }
  };

  // useEffect(() => {
  //   console.log("State güncellendi:", detectedObjectLocations);
  // }, [detectedObjectLocations]);


  // const handleObjectClick = async (detectedObject) => {
  //   const { obj_id, gps_data, obj_info } = detectedObject;

  //   // Kamera isteği
  //   if (obj_info) {
  //     await requestVideo(obj_info.cam_id);
  //   } else {
  //     document.getElementById("image").src = "";
  //     alert("no camera data");
  //   }

  //   // GPS varsa haritaya uç
  //   if (gps_data && map) {
  //     console.log("zooming to: ", gps_data.latitude, gps_data.longitude);
  //     await map.flyTo([gps_data.latitude, gps_data.longitude], 10);
  //   } else {
  //     alert("gps verisi yok");
  //   }

  //   // Bilgi kutusunu açık tut
  //   setdetectedObjects((prev) => {
  //     const currentObject = prev[obj_id];
  //     return {
  //       ...prev,
  //       [obj_id]: {
  //         ...currentObject,
  //         showInfo: true // her tıklamada açık kalsın
  //       }
  //     };
  //   });
  // };

  const [webSockets, setWebSockets] = useState([]);

  const [detectedShapes, setDetectedShapes] = useState({});

  useEffect(() => {
    const setupSockets = async () => {
      const gpsWs = await generateGpsDataWs(setdetectedObjects, activeTrackIdRef);
      const objInfoWs = await generateObjectInfoWs(setdetectedObjects);
      const videoStreamWs = await generateVideoStreamSocket()
      setWebSockets([gpsWs, objInfoWs, videoStreamWs]);

      
    }

    setupSockets();
    fetchShapeData(setDetectedShapes,drawnItemsRef);
    
    const interval = setInterval(async () => {
      const fetchedGpsData = await fetchGpsData(activeTrackIdRef, setdetectedObjects);
      console.log("fetch: ", activeTrackId);

      setdetectedObjects(prev => {
      const updated = {};
      const activeId = String(activeTrackIdRef?.current || "");

      for (const [obj_id, detectedObject] of Object.entries(prev)) {
        // Eğer aktif takip edilen obje varsa ve bu objeye aitse, olduğu gibi bırak
        if (activeId && obj_id === activeId) {
          updated[obj_id] = detectedObject;
          continue;
        }

        // gps_data yoksa olduğu gibi ekle
        if (!detectedObject.gps_data) {
          updated[obj_id] = detectedObject;
          continue;
        }

        // gps_data varsa ve aktif takip edilen obje değilse, durum güncelle
        let newState = detectedObject.gps_data.recievedState;

        if (newState === 'received') {
          newState = 'waiting';
        } else if (newState === 'waiting') {
          newState = 'not received';
        }

        updated[obj_id] = {
          ...detectedObject,
          gps_data: {
            ...detectedObject.gps_data,
            recievedState: newState
          }
        };
      }

      console.log('Detected objects updated:', updated);
      return updated;
    });

    }, 1000);

    const cameraInterval = setInterval(async () => {
      let asd = await fetchCameraData(setCameraObjects);
      
      console.log("camera: ", asd);
      console.log("camera1: ", cameraObjects);
    }, 5000); // Kamera verisini 5 saniyede bir çekiyoruz

    return () => {
      webSockets.map((ws) => ws.close())
      clearInterval(interval);
      clearInterval(cameraInterval);
    };
  }, []); 

useEffect(() => {
  console.log("cameraObjects güncellendi:", cameraObjects);
}, [cameraObjects]);


useEffect(() => {
  if (activeTrackIdRef.current == null || detectedObjectLocations.length === 0) return;

  const currentPoint = detectedObjectLocations[sliderIndex];
  if (!currentPoint) return;

  const updated = { ...detectedObjectsRef.current };

  updated[activeTrackIdRef.current] = {
    ...updated[activeTrackIdRef.current],
    gps_data: {
      ...(updated[activeTrackIdRef.current]?.gps_data || {}),
      latitude: currentPoint.latitude,
      longitude: currentPoint.longitude,
      last_seen: currentPoint.last_seen
    }
  };

  setdetectedObjects(updated);
}, [sliderIndex, detectedObjectLocations, activeTrackId]); // ✅ bağımlılıklar eklendi





  const dangerInfoListRef = useRef(dangerInfoList);
  const changedItemsRef = useRef(changedItems);
  const detectedObjectsRef = useRef({});
  const detectedObjectLocationsRef = useRef(detectedObjectLocations)
  const mapRef = useRef(null);
  const polylineRef = useRef(null);
  const activeTrackIdRef = useRef(null);
  const iRef = useRef(1);
  const cameraObjectsRef = useRef(cameraObjects);
  useEffect(() => {
    dangerInfoListRef.current = dangerInfoList;
    changedItemsRef.current = changedItems;
    detectedObjectsRef.current = detectedObjects;
    detectedObjectLocationsRef.current = detectedObjectLocations;
    mapRef.current = map;
    activeTrackIdRef.current = activeTrackId;
    cameraObjectsRef.current = cameraObjects;
  }, [dangerInfoList, changedItems, detectedObjects, detectedObjectLocations, activeTrackId, cameraObjects]);


  useEffect(() => {

        if (iRef.current === 1) {
          const history =  fetchGpsDataByID('4');
          setDetectedObjectLocations(history);
          iRef.current = 0;
        }
    const interval = setInterval(async () => {
      checkAnyChanges( dangerInfoListRef.current, setDangerInfoList, setChangedItems, mapRef.current, detectedObjectsRef.current);
      
    }, 1000);

    return () => clearInterval(interval);
  }, []); // ✅ sadece activeTrackId'ye bağlı

  // useEffect(() => {
  //   const asd = fetchGpsDataByID(setDetectedObjectLocations, '4'); 
  //   if (Array.isArray(asd) && asd.length > 0) {
  //     const path = asd.map(loc => [loc.latitude, loc.longitude]);
  //     setActivePolyline(path);
  //     console.log("asdasd: ", activePolyline.length);
  //   }
  // }, [detectedObjectLocations, activePolyline]); 

  const foundIcon = new Icon({
    iconUrl: process.env.PUBLIC_URL + "/images/ship.png",
    iconSize: [38, 38]
  });

  const notFoundIcon = new Icon({
    iconUrl: process.env.PUBLIC_URL + "/images/ship.png",
    iconSize: [38, 38]
  });

  const cameraIcon = new Icon({
    iconUrl: process.env.PUBLIC_URL + "/images/camera.png",
    iconSize: [38, 38]
  });


  const [map, setMap] = useState(null);
      useEffect(() => {
      if (map) {
        console.log("SUCCESS: Map instance received, setting up offline tiles.");
        // ... your existing offline tile layer logic ...
      }
    }, [map]);
  useEffect(() => {
    if (map) {
      // Offline tile layer
      const tileLayerOffline = L.tileLayer.offline(
        'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        {
          attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          subdomains: 'abc',
          minZoom: 6,
          maxZoom: 18,
        }
      );
      tileLayerOffline.addTo(map);

      // Add a control to save tiles for offline use
      const controlSaveTiles = L.control.savetiles(tileLayerOffline, {
        zoomlevels: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        confirm: (layer, successCallback) => {
          if (window.confirm('Save the map tiles for offline use?')) {
            successCallback();
          }
        },
        confirmRemoval: (layer, successCallback) => {
          if (window.confirm('Remove the offline map tiles?')) {
            successCallback();
          }
        },
      });
      controlSaveTiles.addTo(map);

    }
  }, [map]);

  useEffect(() => {
    if (activePolyline.length > 1) {
      console.log("Polyline güncellendi:", activePolyline);
      setIsAny(true);
      // map.fitBounds(L.latLngBounds(activePolyline));
    }
  }, [activePolyline, isAny]);

  // useEffect(() => {
  //   if (!activeTrackId) return;

  //   const interval = setInterval(() => {
  //     drawPolylineForObject(activeTrackId);
  //   }, 1000); 

  //   return () => clearInterval(interval);
  // }, [activeTrackId]);


  useEffect(() => {
    if(map){
      const container = document.querySelector('.object-info-container');
      const videoBox = document.getElementById("videoBox");
      const fullscreenBtn = document.getElementById("fullscreenBtn");

      fullscreenBtn.addEventListener("click", () => {
        if (!document.fullscreenElement) {
          videoBox.requestFullscreen();
        } else {
          document.exitFullscreen();
        }
      });


      const enableMapScroll = () => {
        if (map) map.scrollWheelZoom.enable();
      };

      const disableMapScroll = () => {
        if (map) map.scrollWheelZoom.disable();
      };

      if (container) {
        container.addEventListener('mouseenter', disableMapScroll);
        container.addEventListener('mouseleave', enableMapScroll);
      }

      return () => {
        if (container) {
          container.removeEventListener('mouseenter', disableMapScroll);
          container.removeEventListener('mouseleave', enableMapScroll);
        }
      };
    }
  }, [map]);

  const handleSliderChange = (e) => {
    const index = Number(e.target.value);
    setSliderIndex(index);
  };




  return (
    <div style={{width: '100%', height: '100%'}}>
      <div style={{position: "relative", width: '100%', height: '100%'}}>
        <MapContainer
          center={[39.0, 35.0]} // Centered on Turkey
          zoom={6}
          style={{ height: "95vh", width: "100%", position: "absolute"}}
        >
        <MapInitializer setMap={setMap} drawnItemsRef={drawnItemsRef} setDetectedShapes={setDetectedShapes} />

        {/* The TileLayer is now added via the useEffect hook to enable offline functionality */}
        {/* You can keep a standard TileLayer here as a fallback if you prefer */}
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        {/* GPS verisi olan objeleri Marker olarak göster */}
        {Object.entries(detectedObjects)
        .filter(([obj_id]) => obj_id !== activeTrackId) // aktif olanı listeden çıkar
        .map(([obj_id, detectedObject], i) => {
          const gps = detectedObject.gps_data;
          if (!gps) return null;
          if (gps) {
            return (
              <Marker
                key={`marker-${i}`}
                position={[gps.latitude, gps.longitude]}
                icon={gps.recievedState === 'not received' ? notFoundIcon : foundIcon}
                eventHandlers={{
                  click: async () => {
                    if (activeTrackId === obj_id) {
                      // Aynı objeye tekrar tıklanırsa kapat
                      if (polylineRef.current) {
                        console.log("girdi");
                        map.removeLayer(polylineRef.current);
                        polylineRef.current = null;
                      }
                      setActiveTrackId(null);
                      return;
                    }

                    // Önce eski çizgiyi sil
                    if (polylineRef.current) {
                      map.removeLayer(polylineRef.current);
                      polylineRef.current = null;
                    }

                    setActiveTrackId(obj_id);
                    await drawPolylineForObject(obj_id);
                    console.log("active: ", detectedObjectLocations);
                  }
                }}
              >
              </Marker>
            );
          }
          return null;
        })}
        {activeTrackId &&
          detectedObjectLocations[sliderIndex] && (
            <Marker
              position={[
                detectedObjectLocations[sliderIndex].latitude,
                detectedObjectLocations[sliderIndex].longitude
              ]}
              icon={foundIcon}
              eventHandlers={{popupclose: () => {
                            // Popup kapanınca polyline'ı sil ve objeyi null yap
                            console.log("kapandı: ");
                            if (polylineRef.current) {
                              console.log("kapandı1: ");
                              map.removeLayer(polylineRef.current);
                              polylineRef.current = null;
                            }
                            //setSliderIndex(0);
                            setActiveTrackId(null);
                          }}}
            >
              <Popup autoPan={false} closeOnClick={false} >
                
                <div className="popup-card">
                  <div className="popup-title">
                    <span>📍 Object Location</span>
                  </div>
                  <div className="popup-info-grid">
                    <div><strong>ID</strong></div>
                    <div>{activeTrackId}</div>

                    <div>🌐 <strong>Lat</strong></div>
                    <div>{detectedObjectLocations[sliderIndex].latitude}</div>

                    <div>🌐 <strong>Lng</strong></div>
                    <div>{detectedObjectLocations[sliderIndex].longitude}</div>

                    {detectedObjectLocations[sliderIndex].last_seen && (
                      <>
                        <div>🕒 <strong>Last Seen</strong></div>
                        <div>{detectedObjectLocations[sliderIndex].last_seen}</div>
                      </>
                    )}
                  </div>
                </div>
              </Popup>
            </Marker>
        )}
        {/* Kamera verisi olan objeleri Marker olarak göster */}
        {Object.entries(cameraObjects).map(([id, cameraObj], i) => {
          const camData = cameraObj.camera_data;
          if (!camData) return null;

          return (
            <Marker
              key={`camera-marker-${i}`}
              position={[camData.latitude, camData.longitude]}
              icon={camData.recievedState === 'not received' ? cameraIcon : cameraIcon}
              eventHandlers={{
                click: async () => {
                  await requestVideo(camData.camID);
                }
              }}
            >
              <Popup autoPan={false} closeOnClick={false}>
                <div className="popup-card">
                  <div className="popup-title">
                    <span>📷 Camera Info</span>
                  </div>
                  <div className="popup-info-grid">
                    <div>🎥 <strong>Cam ID</strong></div>
                    <div>{camData.camID}</div>

                    <div>🌐 <strong>Lat</strong></div>
                    <div>{camData.latitude}</div>

                    <div>🌐 <strong>Lng</strong></div>
                    <div>{camData.longitude}</div>
                  </div>
                </div>
              </Popup>
            </Marker>
          );
        })}



          <div className="video-wrapper">
            {activeTrackId && (
              <div
                className="slider-overlay"
                onMouseEnter={() => map?.dragging?.disable()}
                onMouseLeave={() => map?.dragging?.enable()}
              >
                <input
                  type="range"
                  min="0"
                  max={detectedObjectLocations.length - 1}
                  value={sliderIndex}
                  onChange={handleSliderChange}
                  className="video-slider"
                />
              </div>
            )}


            <div className="video-stream-box" id="videoBox">
              <button className="fullscreen-btn" id="fullscreenBtn">⛶</button>
              <img id="image" alt="" />
              <h2 style={{ marginTop: "100px", marginLeft: "50px" }}>
                No frame data to show
              </h2>
            </div>
          </div>


          
        <div className="object-info-container">
          <select
            id="form-select"
            className="form-select"
            aria-label="Tracking Options"
            value={trackingFilter}
            onChange={(e) => setTrackingFilter(e.target.value)}
          >
            <option value="0">Tracking Options</option>
            <option value="1">Camera Stream</option>
            <option value="2">GPS Track</option>
          </select>

          
          {filteredObjects.map((detectedObject, i) => {
          let { obj_id, gps_data, obj_info, showInfo, isFresh } = detectedObject;
          let riskScore = obj_info?.risk_score;
          
          if (trackingFilter === "1") {
            const camId = getLastCamId();
            const isActiveCam = obj_info?.cam_id === camId;

            return (
              <div
                key={i}
                className="object-info"
                style={{ cursor: obj_info ? 'pointer' : 'not-allowed' }}
                onClick={
                  obj_info
                    ? async () => {
                        await requestVideo(obj_info.cam_id);
                      }
                    : undefined
                }
              >
                <div
                  className="object-title"
                  style={{
                    backgroundColor: isActiveCam
                      ? 'rgba(0, 200, 50, 0.4)' // yeşil
                      : 'rgba(200, 0, 50, 0.4)' // kırmızı
                  }}
                >
                  <div>camera id: {obj_info?.cam_id || "N/A"}</div>
                </div>
              </div>
            );
          }
          return (
            <div
              key={i}
              className="object-info"
              style={{
                cursor: map ? 'pointer' : 'not-allowed'
              }}
              onClick={
                map
                  ? async () => {
                      if (obj_info) {
                        await requestVideo(obj_info.cam_id);
                      } else {
                        document.getElementById("image").src = "";
                        alert("no camera data");
                      }
                      if (gps_data && map) {
                        console.log("zooming to: ", gps_data.latitude, gps_data.longitude);
                        await map.flyTo([gps_data.latitude, gps_data.longitude], 12);
                      } else alert("gps verisi yok");
                    }
                  : undefined
              }
            >
              <div
                className="object-title"
                style={{
                  backgroundColor: isFresh
                    ? 'rgba(0, 200, 50, 0.4)' 
                    : 'rgba(200, 0, 50, 0.4)' 
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  setShowInfoMap((prev) => ({
                    ...prev,
                    [obj_id]: !prev[obj_id]
                  }));
                }}
              >
                <div>id: {obj_id}</div>
                <div>risk score: {riskScore}</div>
              </div>

              {showInfoMap[obj_id] && (
                <div className="infos">
                  <div className="info-card">
                    <h4>Object Info</h4>
                    <ul>
                      {obj_info ? (
                        <>
                          <li><strong>Camera ID:</strong> {obj_info.cam_id}</li>
                          <li><strong>Class Name:</strong> {obj_info.class_name}</li>
                          <li><strong>Velocity:</strong> {obj_info.velocity_mps.speed}</li>
                          <li><strong>Risk Level:</strong> {obj_info.risk_score}</li>
                        </>
                      ) : (
                        <li className="error">📷 Camera stream not found</li>
                      )}
                      {trackingFilter === "2" && gps_data && (
                        <>
                          <li><strong>Latitude:</strong> {gps_data.latitude}</li>
                          <li><strong>Longitude:</strong> {gps_data.longitude}</li>
                        </>
                      )}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          );
        })}
              <ToastContainer position="bottom-right" autoClose={5000} />
          </div>
        </MapContainer>
      </div>
    </div>
  );
}

export default Map;