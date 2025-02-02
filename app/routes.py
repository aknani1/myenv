from flask import render_template, request, jsonify
from app import socketio
from .camera_manager import CameraManager
from .config import DEFAULTS


def init_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')
    @app.route('/api/defaults', methods=['GET'])
    def get_defaults():
        """
        Returns the default camera settings as JSON.
        """
        return jsonify(DEFAULTS), 200
        
    @app.route('/api/hard_reset', methods=['POST'])
    def hard_reset():
        try:
            app.camera_manager.is_streaming = False
            app.camera_manager.reset_to_default()
            app.camera_manager.configure_pipeline()
            return jsonify({"message": "Full system reset complete"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/configure', methods=['POST'])
    def configure():
        """
        Request body:
        {
          "module": "rgb" or "depth",
          "resolution": "1280x720",
          "frame_rate": "30"
        }
        """
        try:
            data = request.json
            module = data.get('module')
            resolution = data.get('resolution')  
            frame_rate = data.get('frame_rate') 

            if not (module and resolution and frame_rate):
                return jsonify({"error": "Missing data"}), 400

            width, height = map(int, resolution.split('x'))
            app.camera_manager.update_resolution_and_fps(module, width, height, int(frame_rate))

            return jsonify({
                "message": f"{module.capitalize()} updated to {resolution} @ {frame_rate} FPS"
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# In routes.py - Replace toggle_metadata_endpoint
    @app.route('/api/set_metadata', methods=['POST'])
    def set_metadata():
        data = request.json
        module = data.get('module')
        state = data.get('state')
        
        if module not in ['rgb', 'depth'] or not isinstance(state, bool):
            return jsonify({"error": "Invalid request"}), 400
            
        app.camera_manager.set_metadata(module, state)
        return jsonify({
            "message": f"{module} metadata {'enabled' if state else 'disabled'}",
            "state": state
        })

    @app.route('/api/exposure', methods=['POST'])
    def update_exposure():
        """
        Request body:
        {
          "module": "rgb" or "depth",
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
        try:
            info = app.camera_manager.get_device_info()
            return jsonify(info), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    #################### Socket.IO ####################
    @socketio.on('connect')
    def handle_connect():
            """Only handles connection - doesn't auto-start stream"""
            print('Client connected')
            try:
                # Verify camera presence
                app.camera_manager.get_device_info()
                socketio.emit('device_status', {'connected': True})
            except RuntimeError as e:
                print(f"Camera error: {str(e)}")
                socketio.emit('device_status', {'connected': False, 'reason': 'camera_disconnected'})

    # Update disconnect handler
    @socketio.on('disconnect')
    def handle_disconnect():
        print("[Socket.IO] Client disconnected.")
        app.camera_manager.reset_software_defaults()
        
    @socketio.on('start_stream')
    def handle_start_stream():
            try:
                print("recieved start_Stream")
                if(app.camera_manager.is_streaming):
                    print("already streaming")
                    return
                app.camera_manager.configure_pipeline()
                socketio.start_background_task(stream_frames)  # spawns the generator
            except Exception as e:
                socketio.emit('device_status', {'connected': False, 'reason': str(e)})

    def stream_frames():
        try:
            for frame in app.camera_manager.generate_frames():
                # If streaming was stopped (is_streaming=False), break
                if not app.camera_manager.is_streaming:
                    break
                socketio.emit('video_frame', frame)
        except Exception as e:
            print(f"Streaming error: {str(e)}")
        finally:
            app.camera_manager.stop_stream()
            
    @socketio.on('stop_stream')
    def handle_stop_stream():
        if app.camera_manager.is_streaming:
            app.camera_manager.stop_stream()
            print("[server] Stopping stream for client.")
