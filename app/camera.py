import pyrealsense2 as rs
import numpy as np
import cv2
import base64
from app import socketio
import time

# Store standard config and pipeline at module level
pipeline = rs.pipeline()
config = rs.config()

# Track streaming state
streaming = {"status": True}
exposure_value = {"status": 80}
# Keep separate current settings for color and depth
# e.g., resolution and fps
current_settings = {
    "color": {
        "width": 640,
        "height": 360,
        "fps": 30
    },
    "depth": {
        "width": 640,
        "height": 360,
        "fps": 30
    }
}

# Track last frame times for displayed FPS calculation
last_color_time = 0.0
last_depth_time = 0.0
displayed_color_fps = 0.0
displayed_depth_fps = 0.0

# Metadata toggles
metadata_toggles = {
    "rgb": False,
    "depth": False
}
AVAILABLE_METADATA = [
    (rs.frame_metadata_value.frame_counter,       "Frame Counter"),
    (rs.frame_metadata_value.sensor_timestamp,    "Sensor Timestamp"),
    (rs.frame_metadata_value.backend_timestamp,   "Backend Timestamp"),
    (rs.frame_metadata_value.actual_fps,          "Actual FPS"),
    (rs.frame_metadata_value.auto_exposure,       "Auto Exposure"),
    (rs.frame_metadata_value.white_balance,       "White Balance"),
    (rs.frame_metadata_value.brightness,          "Brightness"),
    (rs.frame_metadata_value.contrast,            "Contrast"),
    (rs.frame_metadata_value.saturation,          "Saturation"),
    (rs.frame_metadata_value.sharpness,           "Sharpness"),
]


def stop_generating_frames():
    global streaming, pipeline
    streaming["status"] = False
    try:
        pipeline.stop()
    except Exception:
        pass


def configure_pipeline():
    """
    Applies the global current_settings to the pipeline config,
    stopping the pipeline if running, then re-starting it.
    """
    global pipeline, config, streaming

    # If streaming is active, stop first
    if streaming["status"]:
        stop_generating_frames()

    # Clear previous config
    config.disable_all_streams()

    # Enable color stream with the updated resolution/fps
    c = current_settings["color"]
    config.enable_stream(
        rs.stream.color,
        c["width"],
        c["height"],
        rs.format.bgr8,
        c["fps"]
    )

    # Enable depth stream likewise
    d = current_settings["depth"]
    config.enable_stream(
        rs.stream.depth,
        d["width"],
        d["height"],
        rs.format.z16,
        d["fps"]
    )

    profile = pipeline.start(config)
    streaming["status"] = True
    print("[Pipeline] Started with new configuration:", current_settings)
    return profile


def gather_metadata_and_profile_info(frame):
    """
    Gather hardware fps from metadata if available, plus resolution, etc.
    """
    lines = []
    # 1) Possibly read actual_fps if supported
    if frame.supports_frame_metadata(rs.frame_metadata_value.actual_fps):
        hw_fps = frame.get_frame_metadata(rs.frame_metadata_value.actual_fps)
        lines.append(f"Actual FPS: {hw_fps}")

    # 2) Additional metadata fields as desired...
    # e.g., frame_number, sensor_timestamp, etc.
    if frame.supports_frame_metadata(rs.frame_metadata_value.frame_counter):
        fc = frame.get_frame_metadata(rs.frame_metadata_value.frame_counter)
        lines.append(f"Frame Count: {fc}")

    # 3) Profile info
    profile = rs.video_stream_profile(frame.get_profile())
    w = profile.width()
    h = profile.height()
    fmt = profile.format()
    lines.append(f"Resolution: {w}x{h}")
    lines.append(f"Pixel Format: {fmt}")

    return lines

def overlay_in_top_left(image, lines, text_color=(0, 255, 0)):
    """
    Draws a bounding box of text lines in the top-left corner
    without letting it go offscreen.
    """
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 25
    margin = 5

    # Determine bounding box size
    max_width = 0
    for line in lines:
        (text_size, _) = cv2.getTextSize(line, font, font_scale, thickness)
        if text_size[0] > max_width:
            max_width = text_size[0]

    box_width = max_width + (margin * 2)
    box_height = (line_height * len(lines)) + (margin * 2)

    x = 150
    y = 15

    # Clamping
    if x + box_width > image.shape[1]:
        x = max(0, image.shape[1] - box_width - 10)
    if y + box_height > image.shape[0]:
        y = max(0, image.shape[0] - box_height - 10)

    
    text_y = y + margin + line_height - 5
    for line in lines:
        cv2.putText(
            image,
            line,
            (x + margin, text_y),
            font,
            font_scale,
            text_color,
            thickness
        )
        text_y += line_height
        
        
        
def generate_frames():
    """
    Continuously yield frames from the pipeline with metadata overlays,
    plus displayed FPS.
    """
    global last_color_time, displayed_color_fps
    global last_depth_time, displayed_depth_fps
    global color_sensor

    profile = configure_pipeline()
    device = profile.get_device()
    sensors = device.query_sensors()

    for sensor in sensors:
        if sensor.get_info(rs.camera_info.name) == "RGB Camera":
            color_sensor = sensor
            break

    if color_sensor is None:
        raise RuntimeError("Color sensor not found!")

    # Step 4: Disable auto-exposure
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)

    # Step 5: Set manual exposure (e.g., 1000 microseconds)
    #exposure_value = 1000  # Adjust this value as needed
    print(f"Manual exposure set to {exposure_value} microseconds.")


    try:
        while streaming["status"]:
            # Wait for frames; if pipeline is stopped externally, it might error
            try:
                frameset = pipeline.wait_for_frames()
            except Exception as e:
                print("[Error waiting for frames]:", e)
                break

            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Compute displayed FPS color
            now_c = time.time()
            if last_color_time != 0:
                dt_c = now_c - last_color_time
                if dt_c > 0:
                    displayed_color_fps = 1.0 / dt_c
            last_color_time = now_c

            # Compute displayed FPS depth
            now_d = time.time()
            if last_depth_time != 0:
                dt_d = now_d - last_depth_time
                if dt_d > 0:
                    displayed_depth_fps = 1.0 / dt_d
            last_depth_time = now_d

            color_image = np.asanyarray(color_frame.get_data())
            depth_colorized_frame = rs.colorizer().colorize(depth_frame)
            depth_image = np.asanyarray(depth_colorized_frame.get_data())

            # If metadata is toggled, gather lines & overlay
            if metadata_toggles["rgb"]:
                lines_rgb = gather_metadata_and_profile_info(color_frame)
                lines_rgb.append(f"Displayed FPS: {displayed_color_fps:.1f}")
                overlay_in_top_left(color_image, lines_rgb, text_color=(0, 255, 0))

            if metadata_toggles["depth"]:
                lines_depth = gather_metadata_and_profile_info(depth_frame)
                lines_depth.append(f"Displayed FPS: {displayed_depth_fps:.1f}")
                overlay_in_top_left(depth_image, lines_depth, text_color=(255, 0, 0))

            _, color_buf = cv2.imencode('.jpg', color_image)
            _, depth_buf = cv2.imencode('.jpg', depth_image)
            color_frame_encoded = base64.b64encode(color_buf).decode('utf-8')
            depth_frame_encoded = base64.b64encode(depth_buf).decode('utf-8')

            yield {"color": color_frame_encoded, "depth": depth_frame_encoded}
        stop_generating_frames()
    finally:
        stop_generating_frames()
        
def change_exposure():
    color_sensor.set_option(rs.option.exposure, exposure_value["status"])


def toggle_metadata(module):
    if module in metadata_toggles:
        metadata_toggles[module] = not metadata_toggles[module]