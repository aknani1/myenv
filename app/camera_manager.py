from .metadata_helpers import gather_metadata_and_profile_info
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import base64
import socketio
import traceback  # for more detailed logging
from .config import DEFAULTS
from collections import deque

class CameraManager:
    """
    Manages Intel RealSense pipeline, configuration, and streaming for RGB & depth.
    """

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.default_settings = DEFAULTS
        self.color_frame_times = deque(maxlen=10)
        self.depth_frame_times = deque(maxlen=10)

        self.current_settings = {
            "color": {
                "width": self.default_settings["color"]["width"],
                "height": self.default_settings["color"]["height"],
                "fps": self.default_settings["color"]["fps"]
            },
            "depth": {
                "width": self.default_settings["depth"]["width"],
                "height": self.default_settings["depth"]["height"],
                "fps": self.default_settings["depth"]["fps"]
            }
        }

        # Device connection status
        self.device_connected = False

        # Streaming state
        self.is_streaming = False

        # Metadata toggles
        self.metadata_toggles = {"rgb": False, "depth": False}

        # Exposure values
        self.exposure_value_rgb = self.default_settings["color"]["exposure"]
        self.exposure_value_depth = self.default_settings["depth"]["exposure"]

        # FPS tracking
        self.last_color_time = 0.0
        self.last_depth_time = 0.0
        self.displayed_color_fps = 0.0
        self.displayed_depth_fps = 0.0

        # Sensor references
        self.color_sensor = None
        self.depth_sensor = None
    def get_device_info(self):
        """
        Returns basic info about the connected RealSense device.
        If pipeline isn't running, attempt to start it so we can query device info.
        """
        if not self.is_streaming:
            try:
                self.configure_pipeline()
            except Exception as e:
                print("[CameraManager] Could not start pipeline for get_device_info:", e)
                raise e  # or handle differently

        try:
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            info = {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
            }
            self.device_connected = True
            return info
        except RuntimeError as e:
            print("[CameraManager] get_active_profile() error:", e)
            raise

    def reset_to_default(self):
        """Full hardware-aware reset"""
        self.stop_stream()
        try:
            # Hardware-level reset
            ctx = rs.context()
            if ctx.query_devices().size() > 0:
                ctx.remove_device(ctx.query_devices()[0].get_info(rs.camera_info.serial_number))
        except:
            pass
        
        # Reinitialize fresh
        self.__init__()
        print("[CameraManager] Full hardware reset complete")
    
    def configure_pipeline(self):
        if self.is_streaming:
            self.stop_stream()

            
        self.config.disable_all_streams()
        self.color_frame_times.clear()
        self.depth_frame_times.clear()
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

        try:
            profile = self.pipeline.start(self.config)
            self.is_streaming = True
            self.device_connected = True
            print("[CameraManager] Pipeline started with:", self.current_settings)
            self._cache_sensors(profile)
        except Exception as e:
            self.device_connected = False
            self.is_streaming = False
            print(f"[CameraManager] Failed to start pipeline: {e}")
            raise

    def _cache_sensors(self, profile: rs.pipeline_profile):
        """
        Store references to RGB and Depth sensors once pipeline is started.
        """
        device = profile.get_device()
        sensors = device.query_sensors()
        for sensor in sensors:
            name = sensor.get_info(rs.camera_info.name).lower()
            if "rgb" in name:
                self.color_sensor = sensor
            elif "depth" in name:
                self.depth_sensor = sensor

        # Disable auto-exposure for color
        if self.color_sensor is not None:
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.color_sensor.set_option(rs.option.exposure, self.exposure_value_rgb)

        # Disable auto-exposure for depth
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
            pass  # Already stopped

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

        self.configure_pipeline()

    def set_metadata(self, module: str, state: bool):
        """Set metadata overlay state for specified module"""
        if module in self.metadata_toggles:
            self.metadata_toggles[module] = state
        print(f"[CameraManager] Metadata for {module} set to {state}")

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
            Generator that yields frames in real time. If the camera
            accidentally disconnects (e.g., cable pulled), we detect it
            and emit server->client events via Socket.IO.
            """
            # Ensure streaming is started
            if not self.is_streaming:
                self.configure_pipeline()

            try:
                while self.is_streaming:
                    try:
                        frameset = self.pipeline.wait_for_frames()
                    except RuntimeError as e:
                        # This is raised if frames no longer arrive; camera offline?
                        if "Frame didn't arrive within 5000" in str(e):
                            print("[CameraManager] Physical camera disconnection detected!")
                            self.device_connected = False
                            # Stop streaming so we don't keep spamming errors
                            self.stop_stream()

                            # Emit a status event with reason
                            socketio.emit('device_status', {
                                'connected': False,
                                'reason': 'camera_disconnected'
                            })
                            break
                        else:
                            print("[CameraManager] Other runtime error while waiting for frames:")
                            traceback.print_exc()
                            raise e

                    color_frame = frameset.get_color_frame()
                    depth_frame = frameset.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                    # Compute displayed FPS (Color)
                    now_c = time.time()
                    if self.last_color_time != 0:
                        dt_c = now_c - self.last_color_time
                        if dt_c > 0:
                            self.color_frame_times.append(dt_c)
                            avg_dt_c = sum(self.color_frame_times) / len(self.color_frame_times)
                            self.displayed_color_fps = 1.0 / avg_dt_c
                    self.last_color_time = now_c

                    # Compute displayed FPS (Depth)
                    now_d = time.time()
                    if self.last_depth_time != 0:
                        dt_d = now_d - self.last_depth_time
                        if dt_d > 0:
                            self.depth_frame_times.append(dt_c)
                            avg_dt_d = sum(self.depth_frame_times) / len(self.depth_frame_times)
                            self.displayed_depth_fps = 1.0 / avg_dt_d
                    self.last_depth_time = now_d

                    # Convert frames to JPG+base64
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())

                    _, color_buf = cv2.imencode(".jpg", color_image)
                    _, depth_buf = cv2.imencode(".jpg", depth_image)
                    color_encoded = base64.b64encode(color_buf).decode("utf-8")
                    depth_encoded = base64.b64encode(depth_buf).decode("utf-8")

                    # Gather metadata if toggled
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
                print("[CameraManager] Exception during streaming:")
                traceback.print_exc()
                self.device_connected = False
                self.stop_stream()
            finally:
                self.stop_stream()
