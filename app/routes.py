from flask import render_template, current_app,request,jsonify
from app import socketio
import pyrealsense2 as rs
from .camera import (
    configure_pipeline, generate_frames, toggle_metadata,
    stop_generating_frames, streaming, current_settings,exposure_value,change_exposure
)


def init_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')
    @app.route('/api/configure', methods=['POST'])
    def configure():
        """
        Example: { "module": "rgb", "resolution": "1280x720", "frame_rate": 15 }
        """
        try:
            data = request.json
            module = data.get('module')  # "rgb" or "depth"
            resolution = data.get('resolution')  # e.g. "1280x720"
            frame_rate = data.get('frame_rate')  # e.g. 15
            if not module or not resolution or not frame_rate:
                return jsonify({"error": "Missing data"}), 400

            width, height = map(int, resolution.split('x'))

            if module == "rgb":
                current_settings["color"]["width"] = width
                current_settings["color"]["height"] = height
                current_settings["color"]["fps"] = int(frame_rate)
            elif module == "depth":
                current_settings["depth"]["width"] = width
                current_settings["depth"]["height"] = height
                current_settings["depth"]["fps"] = int(frame_rate)
            else:
                return jsonify({"error": "Invalid module"}), 400

            # Now reconfigure pipeline
            configure_pipeline()
            return jsonify({
                "message": f"{module.capitalize()} updated to {resolution} @ {frame_rate} FPS"
            }), 200

        except Exception as e:
            print("Error in /api/configure:", e)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/toggle_metadata', methods=['POST'])
    def toggle_metadata_endpoint():
        data = request.json
        module = data.get('module')
        if module not in ['rgb', 'depth']:
            return jsonify({"error": "Invalid module"}), 400
        
        toggle_metadata(module)
        return jsonify({"message": f"{module.capitalize()} metadata toggled", "status": toggle_metadata[module]})

      
        
        
    @app.route('/api/stop_stream', methods=['POST'])
    def stop_stream():
        stop_generating_frames()
        return jsonify({"message": "Streaming stopped"})
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')

    @socketio.on('start_stream')
    def start_stream():
        print("Received start_stream event from client.")
        for frame in generate_frames():
            socketio.emit('video_frame', frame)


    @app.route('/api/exposure', methods=['POST'])
    def update_exposure():
        try:
            # Parse the JSON payload from the client
            data = request.json
            module = data.get('module')
            req_exposure_value = int(data.get('exposure'))  # Exposure value (e.g., 8500)
            
            print(data)
            # Validate input
            if not module or exposure_value is None:
                return jsonify({"error": "Invalid input"}), 400

            # if module == 'depth':
            #     # Ensure depth sensor exists
            #     if len(sensors) < 1:
            #         return jsonify({"error": "Depth sensor not found"}), 500

            #     depth_sensor = sensors[0]  # Assuming depth is the first sensor
            #     if rs.option.exposure in depth_sensor.get_supported_options():
            #         depth_sensor.set_option(rs.option.exposure, exposure_value)
            #         print(f"Depth Module exposure updated to {exposure_value}")
            #     else:
            #         return jsonify({"error": "Exposure option not supported for Depth Module"}), 400

            if module == 'rgb':
                # Ensure RGB sensor exists
                exposure_value["status"] = req_exposure_value
                change_exposure()

            else:
                return jsonify({"error": "Invalid module"}), 400

            return jsonify({
                "message": f"{module.capitalize()} Module exposure updated",
                "exposure": exposure_value
            }), 200

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": str(e)}), 500 


# Call this function in __init__.py after creating the app
# init_routes(current_app)