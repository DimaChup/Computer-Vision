from pymavlink import mavutil
import time
import math
import sys
import cv2
import numpy as np
from vision import VisionSystem

# ==========================================
#      CALIBRATION SECTION
# ==========================================
CONNECTION_STR = 'tcp:127.0.0.1:5762'
TARGET_ALT = 15.0 # Meters

# 1. Map Configuration
MAP_FILE = "map.jpg"
MAP_WIDTH_METERS = 480.0  # Total width of the map image in meters
REF_LAT = 51.425106  
REF_LON = -2.672257

# 2. Target Size Configuration
TARGET_REAL_RADIUS_M = 0.15 # 15 cm radius for the red dot

# 3. Camera Specs
SENSOR_WIDTH_MM = 5.02; FOCAL_LENGTH_MM = 6.0
IMAGE_W = 640; IMAGE_H = 480
# ==========================================

# --- STATES ---
class State:
    INIT = "INIT"
    CONNECTING = "CONNECTING"
    ARMING = "ARMING"
    TAKEOFF = "TAKEOFF"
    SEARCH = "SEARCH"
    CENTERING = "CENTERING" 
    HOVER = "HOVER"         
    APPROACH = "APPROACH"   
    LANDING = "LANDING"     
    DONE = "DONE"

class VisualFlightMission:
    def __init__(self):
        # 1. Load Map
        self.full_map = cv2.imread(MAP_FILE)
        if self.full_map is None:
            print(f"Error: '{MAP_FILE}' not found. Creating a fake green map.")
            self.full_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
            self.full_map[:] = (34, 139, 34)
            cv2.line(self.full_map, (0,0), (1000,1000), (255,255,255), 2)
        
        self.map_h, self.map_w = self.full_map.shape[:2]
        self.pix_per_m = self.map_w / MAP_WIDTH_METERS
        
        # --- CALCULATE DOT SIZE ---
        # Convert 15cm (0.15m) to pixels based on map scale
        raw_radius = TARGET_REAL_RADIUS_M * self.pix_per_m
        # Ensure it's at least 2 pixels so it's visible. Change '2' to '5' if you want it bigger visually.
        self.target_radius_px = max(2, int(raw_radius))
        print(f"Map Scale: 1 meter = {self.pix_per_m:.2f} pixels")
        print(f"Target Size: {TARGET_REAL_RADIUS_M}m = {self.target_radius_px} pixels radius")
        
        # 2. Interactive Target Placement
        self.select_target_on_map()
        
        # 3. Vision System
        self.eyes = VisionSystem(camera_index=None)
        
        # 4. Flight State
        self.state = State.INIT
        self.master = None
        self.last_req = 0
        self.last_heartbeat = 0
        
        # 5. Telemetry
        self.lat = REF_LAT
        self.lon = REF_LON
        self.alt = 0.0
        self.roll = 0; self.pitch = 0; self.yaw = 0
        
        # 6. Mission Data
        self.waypoints = []
        self.wp_index = 0
        self.target_lat = 0
        self.target_lon = 0
        self.landing_lat = 0
        self.landing_lon = 0
        self.final_dist = 0.0
        
        # View Params
        self.view_w_px = 100
        self.view_h_px = 100

    def select_target_on_map(self):
        # Resize for display if map is too big to fit screen
        h, w = self.full_map.shape[:2]
        MAX_DISPLAY_H = 800  # Max height in pixels
        
        scale_factor = 1.0
        if h > MAX_DISPLAY_H:
            scale_factor = MAX_DISPLAY_H / h
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            display_map = cv2.resize(self.full_map, (new_w, new_h))
        else:
            display_map = self.full_map.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                temp_vis = display_map.copy()
                # Scale visual radius for the smaller map
                vis_radius = max(2, int(self.target_radius_px * scale_factor))
                cv2.circle(temp_vis, (x, y), vis_radius, (0, 0, 255), -1)
                cv2.imshow("Select Target", temp_vis)
                
                # Calculate REAL coordinates on full resolution map
                real_x = int(x / scale_factor)
                real_y = int(y / scale_factor)
                self.target_px = (real_x, real_y)

        cv2.imshow("Select Target", display_map)
        cv2.setMouseCallback("Select Target", mouse_callback)
        print("CLICK map to place dummy. Press KEY to start mission.")
        cv2.waitKey(0)
        cv2.destroyWindow("Select Target")
        
        if hasattr(self, 'target_px'):
            # Draw permanent target on map
            cv2.circle(self.full_map, self.target_px, self.target_radius_px, (0, 0, 255), -1)

    def gps_to_pixels(self, lat, lon):
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
        lon_m = 111132.954 * math.cos(math.radians(lat))
        
        dy = -(lat - REF_LAT) * lat_m
        dx = (lon - REF_LON) * lon_m
        
        return int(dx * self.pix_per_m), int(dy * self.pix_per_m)

    def get_drone_view(self, cx, cy):
        fov = 2 * math.atan(SENSOR_WIDTH_MM / (2 * FOCAL_LENGTH_MM))
        safe_alt = max(1.0, self.alt)
        ground_w = 2 * safe_alt * math.tan(fov / 2)
        
        self.view_w_px = int(ground_w * self.pix_per_m)
        self.view_h_px = int(self.view_w_px * (IMAGE_H / IMAGE_W))
        
        M = cv2.getRotationMatrix2D((cx, cy), math.degrees(self.yaw), 1.0)
        rot_map = cv2.warpAffine(self.full_map, M, (self.map_w, self.map_h))
        
        x1 = cx - self.view_w_px // 2; y1 = cy - self.view_h_px // 2
        
        crop = np.zeros((self.view_h_px, self.view_w_px, 3), dtype=np.uint8)
        
        y1s, y2s = max(0, y1), min(self.map_h, y1+self.view_h_px)
        x1s, x2s = max(0, x1), min(self.map_w, x1+self.view_w_px)
        y1d, y2d = max(0, -y1), max(0, -y1) + (y2s - y1s)
        x1d, x2d = max(0, -x1), max(0, -x1) + (x2s - x1s)

        if y2s > y1s and x2s > x1s:
            crop[y1d:y2d, x1d:x2d] = rot_map[y1s:y2s, x1s:x2s]
            return cv2.resize(crop, (IMAGE_W, IMAGE_H))
        
        return np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)

    def get_god_view(self, cx, cy):
        display_map = self.full_map.copy()
        cv2.circle(display_map, (cx, cy), 8, (255, 0, 0), -1)
        rect = ((cx, cy), (self.view_w_px, self.view_h_px), math.degrees(self.yaw))
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(display_map, [box], 0, (0, 255, 255), 2)
        
        # Draw calculated target if available
        if self.target_lat != 0:
            tx, ty = self.gps_to_pixels(self.target_lat, self.target_lon)
            cv2.circle(display_map, (tx, ty), 5, (0, 255, 0), -1) # Green for detected target
        
        # Draw Landing Spot
        if self.landing_lat != 0:
            lx, ly = self.gps_to_pixels(self.landing_lat, self.landing_lon)
            cv2.circle(display_map, (lx, ly), 5, (255, 0, 255), -1) # Magenta for landing
            cv2.circle(display_map, (lx, ly), 20, (255, 255, 255), 1) # Ring

        target_h = IMAGE_H
        scale = target_h / self.map_h
        return cv2.resize(display_map, (int(self.map_w * scale), target_h))

    def update_dashboard(self):
        """ Updates the GUI and Returns True if target is seen """
        px, py = self.gps_to_pixels(self.lat, self.lon)
        
        drone_frame = self.get_drone_view(px, py)
        god_frame = self.get_god_view(px, py)
        
        # 1. Draw Crosshair on Drone View (Center Indicator)
        cx, cy = IMAGE_W // 2, IMAGE_H // 2
        cv2.line(drone_frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 0), 2) # Cyan Horizontal
        cv2.line(drone_frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 0), 2) # Cyan Vertical
        
        # 2. Vision Logic
        found, u, v = self.eyes.process_frame_manually(drone_frame)
        if found:
            cv2.circle(drone_frame, (u, v), 10, (0, 255, 0), 2)
            cv2.line(drone_frame, (u, v), (cx, cy), (0, 255, 0), 1) # Line from target to center
            cv2.putText(drone_frame, "TARGET ACQUIRED", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Dashboard Text
        cv2.putText(drone_frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(drone_frame, f"Alt: {self.alt:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 2. Target Info (Green)
        if self.target_lat != 0:
            cv2.putText(drone_frame, f"Tgt GPS: {self.target_lat:.6f}, {self.target_lon:.6f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # 3. Landing Info (Magenta)
        if self.landing_lat != 0:
            cv2.putText(drone_frame, f"Land GPS: {self.landing_lat:.6f}, {self.landing_lon:.6f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 4. Final Result (White)
        if self.state == State.DONE:
             cv2.putText(drone_frame, f"FINAL LANDING: {self.final_dist:.2f}m from Target", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined = np.hstack((god_frame, drone_frame))
        cv2.imshow("Mission Dashboard", combined)
        
        return found, u, v

    def update_telemetry(self):
        if not self.master: return
        while True:
            msg = self.master.recv_match(blocking=False)
            if not msg: break
            
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.alt = msg.relative_alt / 1000.0
            elif msg.get_type() == 'ATTITUDE':
                self.roll = msg.roll
                self.pitch = msg.pitch
                self.yaw = msg.yaw
            elif msg.get_type() == 'HEARTBEAT':
                self.last_heartbeat = time.time()

    def generate_search_pattern(self):
        print("Generating Search Grid...")
        SWATH = 20.0; WIDTH = 120.0; HEIGHT = 120.0
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(self.lat))
        lon_m = 111132.954 * math.cos(math.radians(self.lat))
        
        start_lat = self.lat; start_lon = self.lon
        wps = []
        num_legs = int(HEIGHT / SWATH)
        for i in range(num_legs):
            y_offset = i * SWATH
            if i % 2 == 0: p1_x, p1_y, p2_x, p2_y = 0, y_offset, WIDTH, y_offset
            else: p1_x, p1_y, p2_x, p2_y = WIDTH, y_offset, 0, y_offset
            
            wp1 = (start_lat - (p1_y/lat_m), start_lon + (p1_x/lon_m))
            wp2 = (start_lat - (p2_y/lat_m), start_lon + (p2_x/lon_m))
            wps.append(wp1); wps.append(wp2)
        self.waypoints = wps

    def calculate_target_gps(self, u, v):
        Cx = IMAGE_W / 2; Cy = IMAGE_H / 2
        gsd_m = (SENSOR_WIDTH_MM * self.alt) / (FOCAL_LENGTH_MM * IMAGE_W)
        delta_x_px = u - Cx 
        delta_y_px = v - Cy 
        
        delta_x_m = delta_x_px * gsd_m
        delta_y_m = delta_y_px * gsd_m
        
        fwd_m = -delta_y_m
        right_m = delta_x_m
        
        offset_n = fwd_m * math.cos(self.yaw) - right_m * math.sin(self.yaw)
        offset_e = fwd_m * math.sin(self.yaw) + right_m * math.cos(self.yaw)
        
        R_EARTH = 6378137.0
        dLat = (offset_n / R_EARTH) * (180 / math.pi)
        dLon = (offset_e / (R_EARTH * math.cos(math.radians(self.lat)))) * (180 / math.pi)
        
        self.target_lat = self.lat + dLat
        self.target_lon = self.lon + dLon
        print(f"Target Located at: {self.target_lat:.6f}, {self.target_lon:.6f}")

    def run(self):
        print("Starting Mission...")
        while True:
            self.update_telemetry()
            target_found, px_u, px_v = self.update_dashboard()

            # --- FLIGHT STATE MACHINE ---
            if self.state == State.INIT:
                if time.time() - self.last_req > 1.0:
                    print(f"[{self.state}] Connecting to {CONNECTION_STR}...")
                    try:
                        self.master = mavutil.mavlink_connection(CONNECTION_STR)
                        self.state = State.CONNECTING
                    except: pass
                    self.last_req = time.time()

            elif self.state == State.CONNECTING:
                if self.last_heartbeat > 0:
                    print(f"[{self.state}] Connected! Requesting Data...")
                    self.master.mav.request_data_stream_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
                    )
                    self.state = State.ARMING
                else:
                    if time.time() - self.last_req > 1.0:
                        print(".", end="", flush=True)
                        self.last_req = time.time()

            elif self.state == State.ARMING:
                if time.time() - self.last_req > 2.0:
                    print(f"\n[{self.state}] Arming...")
                    mode_id = self.master.mode_mapping()['GUIDED']
                    self.master.mav.set_mode_send(
                        self.master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()

                if self.master.motors_armed():
                    print(f"\n[{self.state}] ARMED! Taking Off.")
                    self.state = State.TAKEOFF
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, TARGET_ALT)

            elif self.state == State.TAKEOFF:
                if self.alt >= TARGET_ALT * 0.90:
                    print(f"\n[{self.state}] Altitude Reached. Starting Search.")
                    self.generate_search_pattern()
                    self.state = State.SEARCH
                elif time.time() - self.last_req > 5.0:
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, TARGET_ALT)
                    self.last_req = time.time()

            elif self.state == State.SEARCH:
                if target_found:
                    print(f"\n[{self.state}] !!! TARGET DETECTED !!! Stopping to Align.")
                    self.calculate_target_gps(px_u, px_v)
                    self.state = State.CENTERING
                    continue

                if self.wp_index < len(self.waypoints):
                    target = self.waypoints[self.wp_index]
                    if time.time() - self.last_req > 2.0:
                        self.master.mav.set_position_target_global_int_send(
                            0, self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                            0b110111111000, 
                            int(target[0] * 1e7), int(target[1] * 1e7), TARGET_ALT,
                            0, 0, 0, 0, 0, 0, 0, 0)
                        self.last_req = time.time()
                    
                    lat_scale = 111132.0 
                    dist = math.sqrt(((self.lat-target[0])*lat_scale)**2 + ((self.lon-target[1])*lat_scale*0.62)**2)
                    if dist < 2.0:
                        self.wp_index += 1
                else:
                    print("Search Complete. No Target.")
                    self.state = State.DONE

            elif self.state == State.CENTERING:
                # Continuously update the target GPS based on current view 
                if target_found:
                    self.calculate_target_gps(px_u, px_v)
                
                # Send Command to go exactly to Target Lat/Lon
                if time.time() - self.last_req > 0.5: 
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.target_lat * 1e7), int(self.target_lon * 1e7), TARGET_ALT,
                        0, 0, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()

                # Check if we are directly above (Horizontal Distance < 0.5m)
                lat_scale = 111132.0
                dist = math.sqrt(((self.lat-self.target_lat)*lat_scale)**2 + ((self.lon-self.target_lon)*lat_scale*0.62)**2)
                
                if dist < 0.5:
                    print(f"[{self.state}] ALIGNED (Error: {dist:.2f}m). Hovering for confidence.")
                    self.state = State.HOVER
                    self.last_req = time.time()

            elif self.state == State.HOVER:
                # Wait 5 seconds
                if time.time() - self.last_req > 5.0:
                    print(f"[{self.state}] Target Verified. Calculating Landing Offset.")
                    
                    # Calculate 7.5m Offset North
                    R_EARTH = 6378137.0
                    offset_dist = 7.5
                    offset_deg_lat = (offset_dist / R_EARTH) * (180 / math.pi)
                    
                    self.landing_lat = self.target_lat + offset_deg_lat
                    self.landing_lon = self.target_lon
                    
                    print(f"Landing Target: {self.landing_lat:.6f}, {self.landing_lon:.6f}")
                    self.state = State.APPROACH
                    self.last_req = time.time()
                else:
                    # Hold position
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.target_lat * 1e7), int(self.target_lon * 1e7), TARGET_ALT,
                        0, 0, 0, 0, 0, 0, 0, 0)

            elif self.state == State.APPROACH:
                if time.time() - self.last_req > 2.0:
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.landing_lat * 1e7), int(self.landing_lon * 1e7), TARGET_ALT,
                        0, 0, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()
                
                lat_scale = 111132.0 
                dist = math.sqrt(((self.lat-self.landing_lat)*lat_scale)**2 + ((self.lon-self.landing_lon)*lat_scale*0.62)**2)
                
                if dist < 1.0:
                    print(f"[{self.state}] In Position. Landing.")
                    self.state = State.LANDING

            elif self.state == State.LANDING:
                if self.alt < 0.5:
                    print(f"[{self.state}] Touchdown. Mission Complete.")
                    # Calculate Final Error
                    lat_scale = 111132.0
                    final_error = math.sqrt(((self.lat-self.target_lat)*lat_scale)**2 + ((self.lon-self.target_lon)*lat_scale*0.62)**2)
                    self.final_dist = final_error
                    self.state = State.DONE
                elif time.time() - self.last_req > 1.0:
                    # Descend (0m Alt)
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.landing_lat * 1e7), int(self.landing_lon * 1e7), 0, 
                        0, 0, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()

            if cv2.waitKey(20) & 0xFF == 27: break

if __name__ == "__main__":
    mission = VisualFlightMission()
    mission.run()