import axios from "axios";
import L from "leaflet";

const SHAPE_API_URL = "http://127.0.0.1:8080/api/shape/";

const fetchShapeData = async (setDetectedShapes, drawnItemsRef) => {
  try {
    console.log("Fetching shapes from backend...");
    const response = await axios.get(SHAPE_API_URL);
    console.log("Gelen veri:", response.data);

    const shapes = Array.isArray(response.data)
      ? response.data
      : response.data.data;

    if (!Array.isArray(shapes)) {
      throw new Error("Beklenen şekil listesi bir dizi değil.");
    }

    // 1️⃣ State'i güncelle
    const updated = {};
    shapes.forEach(shape => {
  try {
    const parsedGeoJson = typeof shape.geoJson === "string"
      ? JSON.parse(shape.geoJson)
      : shape.geoJson;

    const geoJsonFeature = parsedGeoJson.type === "Feature"
      ? parsedGeoJson
      : {
          type: "Feature",
          geometry: parsedGeoJson,
          properties: {}
        };

    const layer = L.geoJSON(geoJsonFeature, {
      style: feature => {
        const type = shape.type;
        if (type === "polygon") {
          return { color: "#28a745", weight: 2, fillOpacity: 0.2 };
        } else if (type === "rectangle") {
          return { color: "#dc3545", weight: 2, fillOpacity: 0.2 };
        } else if (type === "circle") {
          return { color: "#007bff", weight: 2, fillOpacity: 0.2 };
        }
        return { color: "#17a2b8", weight: 2, fillOpacity: 0.2 };
      },
      pointToLayer: (feature, latlng) => {
        return L.circleMarker(latlng, {
          radius: 8,
          color: "#4facfe",
          fillColor: "#4facfe",
          fillOpacity: 0.5
        });
      },
      onEachFeature: (feature, layer) => {
        const type = shape.type || "Bilinmiyor";
        const id = shape.id || "Tanımsız";

        layer.bindPopup(`
          <div class="popup-card">
            <div class="popup-title">
              📐 Shape Info
            </div>
            <div class="popup-info-grid">
              <div><strong>Type</strong></div>
              <div>${type}</div>

              <div><strong>ID</strong></div>
              <div>${id}</div>
            </div>
          </div>
        `);
      }
    });

        layer.eachLayer(subLayer => {
          subLayer._shapeId = shape.id;
          if (!isShapeAlreadyDrawn(shape.id, drawnItemsRef)) {
            drawnItemsRef.current.addLayer(subLayer);
          } else {
            console.log(`Shape ${shape.id} zaten eklenmiş, atlanıyor.`);
          }
        });


      } catch (err) {
        console.error("GeoJSON parse hatası:", err, shape.geoJson);
      }
    });

  } catch (err) {
    console.error("Shape fetch error:", err);
  }
};

const isShapeAlreadyDrawn = (shapeId, drawnItemsRef) => {
  let exists = false;
  drawnItemsRef.current.eachLayer(layer => {
    if (layer._shapeId === shapeId) {
      exists = true;
    }
  });
  return exists;
};


export {
  fetchShapeData
};
