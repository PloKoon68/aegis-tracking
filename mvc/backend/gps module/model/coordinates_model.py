from flask_sqlalchemy import SQLAlchemy
from flask import jsonify
from datetime import datetime

db = SQLAlchemy()

class Coordinates(db.Model):
    __tablename__ = 'coordinates'

    obj_id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    altitude = db.Column(db.Float, nullable=False)
    lastSeen = db.Column(db.DateTime, nullable=False)

class CoordinatesModel:
    def __init__(self):
        pass

    def checkConsistency(self, data):
        required_keys = {'obj_id', 'latitude', 'longitude', 'altitude', 'lastSeen'}

        if set(data.keys()) != required_keys:
            return False

        try:
            id = data['obj_id']
            latitude = data['latitude']
            longitude = data['longitude']
            altitude = data['altitude']
            lastSeen = data['lastSeen']

            if not isinstance(id, int):
                return False
            if not isinstance(latitude, (int, float)):
                return False
            if not isinstance(longitude, (int, float)):
                return False
            if not isinstance(altitude, (int, float)):
                return False
            if latitude < -90 or latitude > 90:
                return False
            if longitude < -180 or longitude > 180:
                return False
            if altitude < 0:
                return False
            return True

        except (ValueError, TypeError):
            return False

    def insertCoordinate(self, data):
        try:
            coordinate = Coordinates(
                obj_id=data['obj_id'],
                latitude=data['latitude'],
                longitude=data['longitude'],
                altitude=data['altitude'],
                lastSeen=datetime.fromisoformat(data['lastSeen'])
            )
            db.session.add(coordinate)
            db.session.commit()
            return True
        except Exception as e:
            print(e)
            db.session.rollback()
            return False

    def hasTarget(self, data):
        if not self.checkConsistency(data):
            return jsonify({'error': 'Missing or invalid Attributes'}), 400

        try:
            target = Coordinates.query.filter_by(obj_id=data['obj_id']).first()
            return bool(target)
        except Exception as e:
            return jsonify({"Has target error": str(e)}), 500

    def updateCoordinates(self, data):
        if not self.checkConsistency(data):
            return jsonify({'error': 'Missing or invalid Attribute'}), 400

        try:
            coordinate = Coordinates.query.filter_by(obj_id=data['obj_id']).first()
            if coordinate:
                coordinate.latitude = data['latitude']
                coordinate.longitude = data['longitude']
                coordinate.altitude = data['altitude']
                coordinate.lastSeen = datetime.fromisoformat(data['lastSeen'])
                db.session.commit()
                return True
            return False
        except Exception as e:
            print(e)
            db.session.rollback()
            return False

    def getCoordinates(self):
        try:
            return Coordinates.query.all()
        except Exception as e:
            print(e)
            return []

    def getCoordinatesByObjectIds(self, ids):
        try:
            return Coordinates.query.filter(Coordinates.obj_id.in_(ids)).all()
        except Exception as e:
            print(e)
            return []
