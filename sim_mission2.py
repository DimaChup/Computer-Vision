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

# 1. GEO-REFERENCING (From your latest update)
REF_LAT = 51.4234178
REF_LON = -2.6715506

# 2. Connection
CONNECTION_STR = 'tcp:127.0.0.1:5762'

# Camera Specs
SENSOR_WIDTH_MM = 5.02; FOCAL_LENGTH_MM = 6.0
IMAGE_W = 640; IMAGE_H = 480

class SITLBridge:
    def __init__(self):
        # 1. Load Map
        self.full_map = cv2.imread(MAP_FILE)
        if self.full_map is None:
            print("Error: 'map.jpg' not found. Creating a fake green map.")
            self.full_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
            self.full_map[:] = (34, 139, 34)
            # Add some lines so you can see movement
            cv2.line(self.full_map, (0,0), (1000,1000), (255,255,255), 2)
        
        self.map_h, self.map_w = self.full_map.shape[:2]
        self.pix_per_m = self.map_w / MAP_WIDTH_METERS
        
        # 2. Interactive Target Placement (This opens the setup window)
        print("--- SETUP ---")
        self.select_target_on_map()
        
        # 3. Connect to Mission Planner
        print(f"Connecting to SITL at {CONNECTION_STR}...")
        self.master = mavutil.mavlink_connection(CONNECTION_STR)
        
        print("Waiting for Heartbeat...")
        self.master.wait_heartbeat()
        print("Heartbeat Received! Requesting Data Stream...")
        
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
        )
        
        self.eyes = VisionSystem(camera_index=None)
        
        # State
        self.lat = 0; self.lon = 0; self.alt = 0
        self.roll = 0; self.pitch = 0; self.yaw = 0
        
        # Visualization State
        self.view_w_px = 100 # Default
        self.view_h_px = 100

    def select_target_on_map(self):
        """ Opens a window to let user click the target location """
        temp_map = self.full_map.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Draw red dot on the temporary map for feedback
                temp_vis = temp_map.copy()
                cv2.circle(temp_vis, (x, y), 6, (0, 0, 255), -1)
                cv2.imshow("Select Target", temp_vis)
                # Store coordinates
                self.target_px = (x, y)

        cv2.imshow("Select Target", temp_map)
        cv2.setMouseCallback("Select Target", mouse_callback)
        print("CLICK on map to place target. Press ANY KEY to confirm and start.")
        cv2.waitKey(0)
        cv2.destroyWindow("Select Target")
        
        # Draw the target PERMANENTLY on the main map
        if hasattr(self, 'target_px'):
            cv2.circle(self.full_map, self.target_px, 6, (0, 0, 255), -1)
            print(f"Target placed at pixels: {self.target_px}")
        else:
            print("No target selected.")

    def gps_to_pixels(self, lat, lon):
        """ Converts GPS to X/Y pixels on the map image """
        lat_m_per_deg = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
        lon_m_per_deg = 111132.954 * math.cos(math.radians(lat))
        
        delta_lat = lat - REF_LAT
        delta_lon = lon - REF_LON
        
        meters_y = -(delta_lat * lat_m_per_deg) 
        meters_x = delta_lon * lon_m_per_deg
        
        px = int(meters_x * self.pix_per_m)
        py = int(meters_y * self.pix_per_m)
        return px, py

    def get_drone_view(self, cx, cy):
        """ Generates the Camera Feed (Right Side) """
        fov_rad = 2 * math.atan(SENSOR_WIDTH_MM / (2 * FOCAL_LENGTH_MM))
        safe_alt = max(1.0, self.alt)
        ground_w = 2 * safe_alt * math.tan(fov_rad / 2)
        
        # Store these for the God View to use later
        self.view_w_px = int(ground_w * self.pix_per_m)
        self.view_h_px = int(self.view_w_px * (IMAGE_H / IMAGE_W))
        
        # MAVLink Yaw is Clockwise. OpenCV Rotation is Counter-Clockwise.
        # Positive Yaw (Right turn) -> Positive Angle (CCW Rotate Map) -> Correct View
        M = cv2.getRotationMatrix2D((cx, cy), math.degrees(self.yaw), 1.0)
        rot_map = cv2.warpAffine(self.full_map, M, (self.map_w, self.map_h))
        
        x1 = cx - self.view_w_px // 2
        y1 = cy - self.view_h_px // 2
        
        crop = np.zeros((self.view_h_px, self.view_w_px, 3), dtype=np.uint8)
        
        # Bounds checks
        y1s, y2s = max(0, y1), min(self.map_h, y1+self.view_h_px)
        x1s, x2s = max(0, x1), min(self.map_w, x1+self.view_w_px)
        y1d, y2d = max(0, -y1), max(0, -y1) + (y2s - y1s)
        x1d, x2d = max(0, -x1), max(0, -x1) + (x2s - x1s)

        if y2s > y1s and x2s > x1s:
            crop[y1d:y2d, x1d:x2d] = rot_map[y1s:y2s, x1s:x2s]

        return cv2.resize(crop, (IMAGE_W, IMAGE_H))

    def get_god_view(self, cx, cy):
        """ Generates the Map View with Overlays (Left Side) """
        display_map = self.full_map.copy()
        
        # 1. Draw Drone Center
        cv2.circle(display_map, (cx, cy), 8, (255, 0, 0), -1) # Blue dot
        
        # 2. Draw Camera Footprint (Yellow Box)
        # We use the dimensions calculated in get_drone_view
        rect = ((cx, cy), (self.view_w_px, self.view_h_px), math.degrees(self.yaw))
        box = cv2.boxPoints(rect) 
        box = np.int32(box)
        cv2.drawContours(display_map, [box], 0, (0, 255, 255), 2)
        
        # 3. Resize to match the drone view height for clean UI
        target_h = IMAGE_H
        scale = target_h / self.map_h
        target_w = int(self.map_w * scale)
        return cv2.resize(display_map, (target_w, target_h))

    def run(self):
        print("BRIDGE RUNNING.")
        print("Fly the drone in Mission Planner!")
        
        while True:
            # 1. Process ALL available messages in buffer
            # This fixes the issue where asking for GPS discards Attitude packets
            while True:
                msg = self.master.recv_match(blocking=False)
                if not msg:
                    break
                
                msg_type = msg.get_type()
                if msg_type == 'GLOBAL_POSITION_INT':
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt = msg.relative_alt / 1000.0
                elif msg_type == 'ATTITUDE':
                    self.roll = msg.roll
                    self.pitch = msg.pitch
                    self.yaw = msg.yaw

            # 2. Calculate Position
            px, py = self.gps_to_pixels(self.lat, self.lon)
            
            # 3. Generate Views
            drone_frame = self.get_drone_view(px, py) # Must run first to update FOV size
            god_frame = self.get_god_view(px, py)
            
            # 4. Vision
            found, u, v = self.eyes.process_frame_manually(drone_frame)
            if found:
                cv2.circle(drone_frame, (u, v), 10, (0, 255, 0), 2)
                cv2.putText(drone_frame, "TARGET FOUND", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 5. Telemetry Text
            color = (0, 255, 255) if self.lat != 0 else (0, 0, 255)
            cv2.putText(drone_frame, f"Lat: {self.lat:.6f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(drone_frame, f"Lon: {self.lon:.6f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(drone_frame, f"Alt: {self.alt:.1f}m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(drone_frame, f"Yaw: {math.degrees(self.yaw):.0f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 6. Combine
            combined = np.hstack((god_frame, drone_frame))
            cv2.imshow("SITL Dashboard", combined)
            
            if cv2.waitKey(10) == 27: break

if __name__ == "__main__":
    SITLBridge().run()