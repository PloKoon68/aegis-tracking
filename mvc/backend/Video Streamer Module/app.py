from flask import Flask, request, jsonify
from streamer import StreamManager

app = Flask(__name__)
stream_manager = StreamManager()

@app.route('/start', methods=['POST'])
def start_stream():
    print("received request: ")
    data = request.get_json()  ## Get JSON data from the request
    video_id = data.get("video_id")     # Extract video_id from the JSON data
    camera=[1, 2, 3, 4]

    if video_id not in camera: # Check if video_id is valid
        return jsonify({"error": "Invalid video_id"}), 400


    
    result = stream_manager.start_stream(video_id)  # Start the stream
    return jsonify({"message": result})


@app.route('/stop', methods=['POST'])
def stop_stream():
    data = request.get_json()
    video_id = data.get("video_id")
    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    result = stream_manager.stop_stream(video_id)
    return jsonify({"message": result})


@app.route('/status', methods=['GET'])
def status():
    print("gettin status")
    return jsonify(stream_manager.get_status())

@app.route('/')
def home():
    return 'Server running now'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)