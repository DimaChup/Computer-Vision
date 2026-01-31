# filename: sitl_bridge.py
import cv2
import numpy as np
import math
import time
from pymavlink import mavutil
from vision import VisionSystem

# --- CONFIGURATION ---
MAP_FILE = "map.jpg"
MAP_WIDTH_METERS = 200.0 

# 1. GEO-REFERENCING
# I have updated these to match your screenshot results.
# This assumes your drone (at this loc) is roughly at the Top-Left of your map image.
REF_LAT = 51.4234178
REF_LON = -2.6715506

# 2. Connection
# We know this port works from your test!
CONNECTION_STR = 'tcp:127.0.0.1:5762'

# Camera Specs
SENSOR_WIDTH_MM = 5.02; FOCAL_LENGTH_MM = 6.0
IMAGE_W = 640; IMAGE_H = 480

class SITLCamera:
    def __init__(self):
        # Load Map
        self.full_map = cv2.imread(MAP_FILE)
        if self.full_map is None:
            print("Error: 'map.jpg' not found. Creating a fake green map.")
            self.full_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
            self.full_map[:] = (34, 139, 34)
        
        self.map_h, self.map_w = self.full_map.shape[:2]
        self.pix_per_m = self.map_w / MAP_WIDTH_METERS
        
        # Connect to Mission Planner
        print(f"Connecting to SITL at {CONNECTION_STR}...")
        self.master = mavutil.mavlink_connection(CONNECTION_STR)
        
        # CRITICAL FIX: Wait for Heartbeat AND Request Data
        print("Waiting for Heartbeat...")
        self.master.wait_heartbeat()
        print("Heartbeat Received! Requesting Data Stream...")
        
        # This tells Mission Planner "Send me everything!"
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1
        )
        
        self.eyes = VisionSystem(camera_index=None)
        
        # State
        self.lat = 0; self.lon = 0; self.alt = 0
        self.roll = 0; self.pitch = 0; self.yaw = 0

    def gps_to_pixels(self, lat, lon):
        """ Converts GPS to X/Y pixels on the map image """
        # Meters per degree calculation
        lat_m_per_deg = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
        lon_m_per_deg = 111132.954 * math.cos(math.radians(lat))
        
        delta_lat = lat - REF_LAT
        delta_lon = lon - REF_LON
        
        # Invert Lat because Image Y goes down
        meters_y = -(delta_lat * lat_m_per_deg) 
        meters_x = delta_lon * lon_m_per_deg
        
        px = int(meters_x * self.pix_per_m)
        py = int(meters_y * self.pix_per_m)
        return px, py

    def get_drone_view(self, cx, cy):
        fov_rad = 2 * math.atan(SENSOR_WIDTH_MM / (2 * FOCAL_LENGTH_MM))
        # Ensure alt is at least 1m to avoid divide by zero/zero size
        safe_alt = max(1.0, self.alt)
        ground_w = 2 * safe_alt * math.tan(fov_rad / 2)
        
        view_w_px = int(ground_w * self.pix_per_m)
        view_h_px = int(view_w_px * (IMAGE_H / IMAGE_W))
        
        M = cv2.getRotationMatrix2D((cx, cy), math.degrees(self.yaw), 1.0)
        rot_map = cv2.warpAffine(self.full_map, M, (self.map_w, self.map_h))
        
        x1 = cx - view_w_px // 2
        y1 = cy - view_h_px // 2
        
        crop = np.zeros((view_h_px, view_w_px, 3), dtype=np.uint8)
        
        y1_src = max(0, y1); y2_src = min(self.map_h, y1 + view_h_px)
        x1_src = max(0, x1); x2_src = min(self.map_w, x1 + view_w_px)
        y1_dst = max(0, -y1); y2_dst = y1_dst + (y2_src - y1_src)
        x1_dst = max(0, -x1); x2_dst = x1_dst + (x2_src - x1_src)

        if y2_src > y1_src and x2_src > x1_src:
            crop[y1_dst:y2_dst, x1_dst:x2_dst] = rot_map[y1_src:y2_src, x1_src:x2_src]

        return cv2.resize(crop, (IMAGE_W, IMAGE_H))

    def run(self):
        print("BRIDGE RUNNING.")
        print("Controls: Fly the drone in Mission Planner!")
        
        while True:
            # 1. Fetch Telemetry
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
            att = self.master.recv_match(type='ATTITUDE', blocking=False)
            
            if msg:
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.alt = msg.relative_alt / 1000.0
            if att:
                self.roll = att.roll
                self.pitch = att.pitch
                self.yaw = att.yaw

            # 2. Update View
            px, py = self.gps_to_pixels(self.lat, self.lon)
            frame = self.get_drone_view(px, py)
            
            # 3. Vision
            found, u, v = self.eyes.process_frame_manually(frame)
            if found:
                cv2.circle(frame, (u, v), 10, (0, 255, 0), 2)
                cv2.putText(frame, "TARGET FOUND", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 4. Telemetry Overlay
            # If coordinates are 0, warn the user
            color = (0, 255, 255) if self.lat != 0 else (0, 0, 255)
            cv2.putText(frame, f"Lat: {self.lat:.6f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Lon: {self.lon:.6f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Alt: {self.alt:.1f}m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow("SITL Camera View", frame)
            if cv2.waitKey(10) == 27: break

if __name__ == "__main__":
    SITLCamera().run()