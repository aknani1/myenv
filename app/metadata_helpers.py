import pyrealsense2 as rs
import cv2

def gather_metadata_and_profile_info(frame):
    lines = []
    # Check and add FPS if available
    if frame.supports_frame_metadata(rs.frame_metadata_value.actual_fps):
        hw_fps = frame.get_frame_metadata(rs.frame_metadata_value.actual_fps)
        lines.append(f"hardware FPS: {hw_fps}")

    # Check frame counter metadata
    if frame.supports_frame_metadata(rs.frame_metadata_value.frame_counter):
        fc = frame.get_frame_metadata(rs.frame_metadata_value.frame_counter)
        lines.append(f"Frame Count: {fc}")

    # Add resolution and pixel format
    profile = rs.video_stream_profile(frame.get_profile())
    w = profile.width()
    h = profile.height()
    fmt = profile.format()
    lines.append(f"Resolution: {w}x{h}")
    lines.append(f"Pixel Format: {fmt}")

    return lines

def overlay_in_top_left(image, lines, text_color=(255, 255, 255), pos=(0, 2), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 18
    
    # Calculate bounding box size based on text metrics
    max_width = 0
    for line in lines:
        (text_size, _) = cv2.getTextSize(line, font, font_scale, thickness)
        if text_size[0] > max_width:
            max_width = text_size[0]
    box_width = max_width + 20
    box_height = line_height * len(lines) + 16
    
    # Draw a semi-transparent rectangle as background
    x, y = pos
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + box_width, y + box_height),
        bg_color,
        -1
    )
    alpha = 0.4  # transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw text lines inside the rectangle
    y_text = y + 12
    for line in lines:
        cv2.putText(image, line, (x + 10, y_text),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
        y_text += line_height
