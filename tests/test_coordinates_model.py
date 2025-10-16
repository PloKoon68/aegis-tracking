import unittest
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mvc', 'backend'))

from model.coordinates_model import CoordinatesModel

class TestCoordinatesModel(unittest.TestCase):
    
    def setUp(self):
        self.model = CoordinatesModel()
        self.valid_data = {
            'id': '123',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'altitude': 10.5
        }
        self.invalid_data = {
            'id': '123',
            'latitude': 40.7128
            # Missing longitude and altitude
        }
    
    def test_check_consistency_with_valid_data(self):
        """Test checkConsistency with valid data"""
        result = self.model.checkConsistency(self.valid_data)
        self.assertTrue(result)

    def test_check_consistency_with_invalid_data(self):
        """Test checkConsistency with invalid data"""
        result = self.model.checkConsistency(self.invalid_data)
        self.assertFalse(result)

    def test_check_consistency_with_empty_data(self):
        """Test checkConsistency with empty data"""
        result = self.model.checkConsistency({})
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()