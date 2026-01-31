# filename: geolocation.py
import numpy as np
import math

# --- CONFIGURATION: Raspberry Pi Global Shutter Camera ---
# Verified Specs: Sensor Width 5.02mm, Lens 6mm, Image 1456x1088
SENSOR_WIDTH_MM = 5.02
FOCAL_LENGTH_MM = 6.0
IMAGE_WIDTH_PX = 1456
IMAGE_HEIGHT_PX = 1088

# Intrinsic Parameters (Pinhole Model)
Cx = IMAGE_WIDTH_PX / 2
Cy = IMAGE_HEIGHT_PX / 2
Fx = (FOCAL_LENGTH_MM * IMAGE_WIDTH_PX) / SENSOR_WIDTH_MM
Fy = Fx 

def pixel_to_gps(u, v, alt_relative, drone_roll_rad, drone_pitch_rad, drone_yaw_rad, drone_lat, drone_lon):
    """
    Converts a pixel (u, v) to a GPS coordinate.
    """
    if alt_relative <= 0.5: 
        return 0.0, 0.0 

    # 1. Pixel to Camera Ray
    # We invert Y because image Y increases downwards
    x_c = (u - Cx) / Fx
    y_c = (v - Cy) / Fy
    z_c = 1.0 
    ray_camera = np.array([x_c, y_c, z_c])

    # 2. Rotation Matrices (Body to NED)
    # Pitch (Y-axis)
    R_pitch = np.array([
        [math.cos(drone_pitch_rad), 0, math.sin(drone_pitch_rad)],
        [0, 1, 0],
        [-math.sin(drone_pitch_rad), 0, math.cos(drone_pitch_rad)]
    ])
    
    # Roll (X-axis)
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(drone_roll_rad), -math.sin(drone_roll_rad)],
        [0, math.sin(drone_roll_rad), math.cos(drone_roll_rad)]
    ])
    
    # Yaw (Z-axis)
    R_yaw = np.array([
        [math.cos(drone_yaw_rad), -math.sin(drone_yaw_rad), 0],
        [math.sin(drone_yaw_rad), math.cos(drone_yaw_rad), 0],
        [0, 0, 1]
    ])

    R_body_to_ned = R_yaw @ R_pitch @ R_roll
    ray_ned = R_body_to_ned @ ray_camera

    # 3. Ray-Plane Intersection
    if ray_ned[2] <= 0:
        return 0.0, 0.0 
        
    scale = alt_relative / ray_ned[2]
    north_offset = ray_ned[0] * scale
    east_offset = ray_ned[1] * scale

    # 4. Meters to GPS
    R_EARTH = 6378137.0
    dLat = (north_offset / R_EARTH) * (180 / math.pi)
    dLon = (east_offset / (R_EARTH * math.cos(math.radians(drone_lat)))) * (180 / math.pi)

    return drone_lat + dLat, drone_lon + dLon