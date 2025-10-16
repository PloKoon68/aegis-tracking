import axios from "axios";
import { toast } from 'react-toastify';

const HTTP_URL = 'http' + process.env.REACT_APP_GPS_SERVICE_URL + 'api/location/locations-inside-shapes';

const fetchLocationsInsideShapes = async () => {
  try {
    console.log("waiting for locations inside shapes...");
    const locations = (await axios.get(HTTP_URL)).data;

    const simplified = locations.map(obj => ({
      shapeId: obj.shapeId,
      locationId: obj.locationId
    }));

    // State'i direkt bu array ile güncelle
    return simplified;
  } catch (err) {
    console.error("Error fetching locations:", err);
    return err;
  }
};

const isEqualList = (listA, listB) => {
  const groupByLocation = list =>
    list.reduce((acc, item) => {
      const locId = item.locationId;
      if (!acc[locId]) acc[locId] = new Set();
      acc[locId].add(item.shapeId);
      return acc;
    }, {});

  const mapA = groupByLocation(listA);
  const mapB = groupByLocation(listB);

  const keysA = Object.keys(mapA);
  const keysB = Object.keys(mapB);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    const setA = mapA[key];
    const setB = mapB[key];
    if (!setB || setA.size !== setB.size) return false;

    for (const val of setA) {
      if (!setB.has(val)) return false;
    }
  }

  return true;
};

const checkAnyChanges = async (dangerInfoList, setDangerInfoList, setChangedItems, map, detectedObjects) => {
  const newData = await fetchLocationsInsideShapes();

  const normalize = data =>
    data.map(item => ({
      locationId: Number(item.locationId),
      shapeId: Number(item.shapeId),
      objId: item.objId
    }));

  const newNormalized = normalize(newData);
  const prevNormalized = normalize(dangerInfoList);

  if (!isEqualList(prevNormalized, newNormalized)) {
    const buildMap = list => {
      const map = new Map();
      list.forEach(item => {
        if (!map.has(item.locationId)) {
          map.set(item.locationId, new Set());
        }
        map.get(item.locationId).add(item.shapeId);
      });
      return map;
    };

    const prevMap = buildMap(prevNormalized);
    const newMap = buildMap(newNormalized);

    const addedItems = [];
    const removedItems = [];

    newNormalized.forEach(item => {
      const prevShapes = prevMap.get(item.locationId);
      if (!prevShapes || !prevShapes.has(item.shapeId)) {
        addedItems.push({ ...item, changeType: 'add' });
      }
    });

    prevNormalized.forEach(item => {
      const newShapes = newMap.get(item.locationId);
      if (!newShapes || !newShapes.has(item.shapeId)) {
        removedItems.push({ ...item, changeType: 'remove' });
      }
    });

    const allChanges = [...addedItems, ...removedItems];

    setDangerInfoList(newData);
    setChangedItems(allChanges);

    allChanges.forEach(item => {
      if (item.changeType === 'add') {
        toast.error(
          `🚨 Obj_ID ${item.locationId} → Shape_ID ${item.shapeId} alanına girdi`,
          {
            onClick: () => {
              console.log("Detected: ",detectedObjects);
              const gps = detectedObjects[item.locationId]?.gps_data;
              if (gps) {
                map.flyTo([gps.latitude, gps.longitude], 8); 
              }
            }
          }
        );

      } else if (item.changeType === 'remove') {
        toast.info(
          `✅ Obj_ID ${item.locationId} → Shape_ID ${item.shapeId} alanından çıktı`,
          {
            onClick: () => {
              const gps = detectedObjects[item.locationId]?.gps_data;
              if (gps) {
                map.flyTo([gps.latitude, gps.longitude], 8
                );
              }
            }
          }
        );
      }
    });
  } else {
    setChangedItems([]);
  }
};



export {
  fetchLocationsInsideShapes, checkAnyChanges
};
