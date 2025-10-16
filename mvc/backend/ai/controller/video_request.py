from flask import Blueprint, jsonify, current_app

video_bp = Blueprint('video_bp', __name__)

@video_bp.record_once
def init(state):
    pass

# @video_bp.route('/video', methods=['POST'])
# def get_video_coordinates():

#     """for ws in current_app.connected_ws_clients:
#         try:
#             ws.send(""""""Send Processed Video Frames"""""")  # Placeholder for actual video frame data

#         except Exception as e:
#             print("WebSocket send error:", e)"""

#     return jsonify({'message': f'Received Video Id:'})

@video_bp.route('/video/<int:cam_id>', methods=['POST'])
def request_camera(cam_id):
    current_app.current_streaming_cam = cam_id 
    print(f"{current_app.current_streaming_cam} geldi")
    return jsonify({'message': f'Received Video Id:'})

@video_bp.route('/video', methods=['GET'])
def get_video_port():
    cam_id = getattr(current_app, 'current_streaming_cam', None)
    if cam_id is not None:
        return jsonify({'active_cam_id': cam_id})
    else:
        return jsonify({'message': 'No active camera'}), 404
