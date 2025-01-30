from flask import render_template, request, jsonify
from app import socketio
# Import the CameraManager
from .camera_manager import CameraManager

def init_routes(app):
    # We assume an instance of CameraManager is attached to the app:
    # app.camera_manager = CameraManager() (in __init__.py)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/configure', methods=['POST'])
    def configure():
        """
        Example request body:
        {
          "module": "rgb",
          "resolution": "1280x720",
          "frame_rate": 15
        }
        """
        try:
            data = request.json
            module = data.get('module')
            resolution = data.get('resolution')  # e.g. "1280x720"
            frame_rate = data.get('frame_rate')  # e.g. 15

            if not (module and resolution and frame_rate):
                return jsonify({"error": "Missing data"}), 400

            width, height = map(int, resolution.split('x'))
            app.camera_manager.update_resolution_and_fps(module, width, height, int(frame_rate))

            return jsonify({
                "message": f"{module.capitalize()} updated to {resolution} @ {frame_rate} FPS"
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/toggle_metadata', methods=['POST'])
    def toggle_metadata_endpoint():
        data = request.json
        module = data.get('module')
        if module not in ['rgb', 'depth']:
            return jsonify({"error": "Invalid module"}), 400

        app.camera_manager.toggle_metadata(module)
        return jsonify({"message": f"{module.capitalize()} metadata toggled"})

    @app.route('/api/exposure', methods=['POST'])
    def update_exposure():
        """
        Example request body:
        {
          "module": "rgb",
          "exposure": 8500
        }
        """
        try:
            data = request.json
            module = data.get('module')
            exposure_value = int(data.get('exposure'))

            app.camera_manager.set_exposure(module, exposure_value)

            return jsonify({
                "message": f"{module.capitalize()} exposure updated",
                "exposure": exposure_value
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/stop_stream', methods=['POST'])
    def stop_stream():
        app.camera_manager.stop_stream()
        return jsonify({"message": "Streaming stopped"})


            
    @app.route('/api/camera_info', methods=['GET'])
    def camera_info():
        """
        Returns device info from the active RealSense camera.
        """
        try:
            info = app.camera_manager.get_device_info()
            print(info)
            return jsonify(info), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        
        
        #############socket io#############
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        try:
            # Check if device is connected and working
            if app.camera_manager.is_streaming:
                socketio.emit('device_status', {'connected': True})
            else:
                # Try to get device info to verify connection
                info = app.camera_manager.get_device_info()
                socketio.emit('device_status', {'connected': True})
        except RuntimeError as e:
            print(f"Device connection error: {str(e)}")
            socketio.emit('device_status', {'connected': False})
        except Exception as e:
            print(f"General connection error: {str(e)}")
            socketio.emit('device_status', {'connected': False})

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected; resetting camera settings to defaults.')
        app.camera_manager.reset_to_default()
        socketio.emit('device_status', {'connected': False})

        
    @socketio.on('start_stream')
    def start_stream():
        print("Received start_stream event from client.")
        for frame in app.camera_manager.generate_frames():
            socketio.emit('video_frame', frame)
            
            
