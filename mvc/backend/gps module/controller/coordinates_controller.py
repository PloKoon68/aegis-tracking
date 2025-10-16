from flask import Blueprint, request, jsonify, current_app
from model.coordinates_model import CoordinatesModel
from datetime import datetime

coordinates_bp = Blueprint('coordinates_bp', __name__)
coordinates_model = CoordinatesModel()

@coordinates_bp.route('/coordinates', methods=['GET'])
def get_all_coordinates():
    try:
        coordinates = coordinates_model.getCoordinates()
        coordinates_list = [
            {
                'obj_id': c.obj_id,
                'latitude': c.latitude,
                'longitude': c.longitude,
                'altitude': c.altitude,
                'lastSeen': c.lastSeen.isoformat()
            }
            for c in coordinates
        ]
        return jsonify(coordinates_list)
    except Exception as e:
        print("Get Coordinates Database Error", e)
        return jsonify({'error': 'Database error', 'details': str(e)}), 500

@coordinates_bp.route('/coordinates/getByObjectIds', methods=['POST'])
def get_coordinates_by_ids():
    try:
        data = request.get_json()
        obj_ids = data['obj_ids']
        coordinates = coordinates_model.getCoordinatesByObjectIds(obj_ids)
        coordinates_list = [
            {
                'obj_id': c.obj_id,
                'latitude': c.latitude,
                'longitude': c.longitude,
                'altitude': c.altitude,
                'lastSeen': c.lastSeen.isoformat()
            }
            for c in coordinates
        ]
        return jsonify(coordinates_list)
    except Exception as e:
        print("Get Coordinates Database Error", e)
        return jsonify({'error': 'Database error', 'details': str(e)}), 500

@coordinates_bp.route('/coordinates/insert', methods=['POST'])
def insert_coordinates():
    data = request.get_json()
    data['lastSeen'] = datetime.now().isoformat()

    # WebSocket yayını (isteğe bağlı)
    try:
        current_app.connected_ws_clients['gps-data'].send(jsonify(data).get_data(as_text=True))
    except Exception as e:
        print("WebSocket send error:", e)

    print("Saving to PostgreSQL...")

    has_target = coordinates_model.hasTarget(data)
    try:
        if has_target is True:
            result = coordinates_model.updateCoordinates(data)
            if result is True:
                return jsonify({'message': 'Coordinate updated successfully!'}), 201
            return jsonify({'error': 'Coordinate update failed'}), 400

        if has_target is False:
            result = coordinates_model.insertCoordinate(data)
            if result is True:
                return jsonify({'message': 'New coordinate was inserted successfully!'}), 201
            return jsonify({'error': 'Coordinate insertion failed'}), 400

        return has_target
    except Exception as e:
        print("Database Error", e)
        return jsonify({'error': 'Database error', 'details': str(e)}), 500
