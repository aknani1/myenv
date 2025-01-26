from typing import Optional, Dict
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import base64

# If you are using a separate metadata_helpers file, import from there:
from .metadata_helpers import gather_metadata_and_profile_info, overlay_in_top_left

class CameraManager:
    """
    Manages Intel RealSense pipeline, configuration, and streaming for RGB & depth.
    """

    def __init__(self):
        # Initialize pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Default settings for color and depth
        self.current_settings = {
            "color": {"width": 640, "height": 360, "fps": 30},
            "depth": {"width": 640, "height": 360, "fps": 30},
        }
        self.default_settings = {
            "color": {"width": 640, "height": 360, "fps": 30},
            "depth": {"width": 640, "height": 360, "fps": 30},
        }

        # Streaming state
        self.is_streaming = False

        # Metadata toggles
        self.metadata_toggles = {"rgb": False, "depth": False}

        # Exposure values
        self.exposure_value_rgb = 300
        self.exposure_value_depth = 300

        # Track last frame times for displayed FPS
        self.last_color_time = 0.0
        self.last_depth_time = 0.0
        self.displayed_color_fps = 0.0
        self.displayed_depth_fps = 0.0

        # Keep references to sensors
        self.color_sensor = None
        self.depth_sensor = None
        



    def reset_to_default(self):
        """
        Stop the pipeline and reset all relevant fields to default values.
        """
        self.stop_stream()

        # Restore default resolution/fps
        self.current_settings = {
            "color": dict(self.default_settings["color"]),
            "depth": dict(self.default_settings["depth"]),
        }

        # Reset toggles
        self.metadata_toggles = {"rgb": False, "depth": False}

        # Reset exposure
        self.exposure_value_rgb = 300
        self.exposure_value_depth = 300
        # At this point, we do not start the pipeline automatically because
        # we might only start streaming once a new client connects
        print("[CameraManager] Reset to default settings.")
        
        
        
        
        
    def configure_pipeline(self):
        """
        Applies the current_settings to self.config, restarts the pipeline.
        """
        if self.is_streaming:
            self.stop_stream()

        # Clear previous config
        self.config.disable_all_streams()

        # Enable color stream
        c = self.current_settings["color"]
        self.config.enable_stream(
            rs.stream.color,
            c["width"], c["height"], rs.format.bgr8, c["fps"]
        )

        # Enable depth stream
        d = self.current_settings["depth"]
        self.config.enable_stream(
            rs.stream.depth,
            d["width"], d["height"], rs.format.z16, d["fps"]
        )

        # Start pipeline
        profile = self.pipeline.start(self.config)
        self.is_streaming = True
        print("[CameraManager] Pipeline started with:", self.current_settings)

        # Cache sensor references
        self._cache_sensors(profile)

    def _cache_sensors(self, profile: rs.pipeline_profile):
        """
        Internal method to store references to RGB and Depth sensors once pipeline is started.
        """
        device = profile.get_device()
        sensors = device.query_sensors()
        for sensor in sensors:
            name = sensor.get_info(rs.camera_info.name).lower()
            if "rgb" in name:
                self.color_sensor = sensor
            elif "depth" in name:
                self.depth_sensor = sensor

        # Optionally disable auto-exposure for color
        if self.color_sensor is not None:
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.color_sensor.set_option(rs.option.exposure, self.exposure_value_rgb)

        if self.depth_sensor is not None:
             self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
             self.depth_sensor.set_option(rs.option.exposure, self.exposure_value_depth)

    def stop_stream(self):
        """
        Stops the pipeline if running.
        """
        self.is_streaming = False
        try:
            self.pipeline.stop()
            print("[CameraManager] Pipeline stopped.")
        except Exception:
            pass  # In case it's already stopped

    def update_resolution_and_fps(self, module: str, width: int, height: int, fps: int):
        """
        Update resolution/fps in current_settings and reconfigure pipeline.
        """
        if module == "rgb":
            self.current_settings["color"].update({"width": width, "height": height, "fps": fps})
        elif module == "depth":
            self.current_settings["depth"].update({"width": width, "height": height, "fps": fps})
        else:
            raise ValueError("Unknown module name")

        # Reconfigure pipeline to apply new settings
        self.configure_pipeline()

    def toggle_metadata(self, module: str):
        """
        Enable/disable metadata overlay for the specified module (rgb or depth).
        """
        if module in self.metadata_toggles:
            self.metadata_toggles[module] = not self.metadata_toggles[module]
        print(f"[CameraManager] Metadata for {module} set to {self.metadata_toggles[module]}")

    def set_exposure(self, module: str, value: int):
        """
        Update exposure for the specified module (rgb or depth).
        """
        if module == "rgb" and self.color_sensor:
            self.exposure_value_rgb = value
            self.color_sensor.set_option(rs.option.exposure, value)
            print(f"[CameraManager] RGB exposure set to {value}")
        elif module == "depth" and self.depth_sensor:
            self.exposure_value_depth = value
            self.depth_sensor.set_option(rs.option.exposure, value)
            print(f"[CameraManager] Depth exposure set to {value}")
        else:
            print(f"[CameraManager] Invalid module or sensor not found for module={module}")

    def generate_frames(self):
        """
        Generator that yields encoded frames (color, depth) in real time.
        Suitable for passing to a WebSocket or Socket.IO 'emit' loop.
        """
        if not self.is_streaming:
                self.configure_pipeline()

        try:
            while self.is_streaming:
                frameset = self.pipeline.wait_for_frames()
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Calculate displayed FPS
                now_c = time.time()
                if self.last_color_time != 0:
                    dt_c = now_c - self.last_color_time
                    if dt_c > 0:
                        self.displayed_color_fps = 1.0 / dt_c
                self.last_color_time = now_c

                now_d = time.time()
                if self.last_depth_time != 0:
                    dt_d = now_d - self.last_depth_time
                    if dt_d > 0:
                        self.displayed_depth_fps = 1.0 / dt_d
                self.last_depth_time = now_d

                # Convert frames to base64
                color_image = np.asanyarray(color_frame.get_data())
                depth_colorized_frame = rs.colorizer().colorize(depth_frame)
                depth_image = np.asanyarray(depth_colorized_frame.get_data())

                _, color_buf = cv2.imencode(".jpg", color_image)
                _, depth_buf = cv2.imencode(".jpg", depth_image)
                color_encoded = base64.b64encode(color_buf).decode("utf-8")
                depth_encoded = base64.b64encode(depth_buf).decode("utf-8")

                # Gather metadata as lists of text lines (if toggled)
                lines_rgb = []
                lines_depth = []
                if self.metadata_toggles["rgb"]:
                    lines_rgb = gather_metadata_and_profile_info(color_frame)
                    lines_rgb.append(f"Displayed FPS: {self.displayed_color_fps:.1f}")

                if self.metadata_toggles["depth"]:
                    lines_depth = gather_metadata_and_profile_info(depth_frame)
                    lines_depth.append(f"Displayed FPS: {self.displayed_depth_fps:.1f}")

                yield {
                    "color": color_encoded,
                    "depth": depth_encoded,
                    "metadata": {
                        "rgb": lines_rgb,
                        "depth": lines_depth
                    }
                }

        except Exception as e:
            print("[CameraManager] Error in generate_frames():", e)
        finally:
            self.stop_stream()


    def get_device_info(self):
        """
        Returns basic info about the connected RealSense device.
        If pipeline isn't running, it starts it so we can query device info.
        """
        if not self.is_streaming:
            self.configure_pipeline()

        profile = self.pipeline.get_active_profile()
        device = profile.get_device()

        info = {
            "name": device.get_info(rs.camera_info.name),
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "firmware_version": device.get_info(rs.camera_info.firmware_version),
            "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
        }
        return info