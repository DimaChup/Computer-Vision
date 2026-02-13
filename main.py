# main.py
from pymavlink import mavutil
import time
import math
import cv2
import numpy as np
import csv
from datetime import datetime
import sys
import os

# --- Detect if display is available (headless Pi vs desktop) ---
HEADLESS = False
try:
    _test_win = "__headless_test__"
    cv2.namedWindow(_test_win)
    cv2.destroyWindow(_test_win)
except cv2.error:
    HEADLESS = True
    print("[INFO] Running in HEADLESS mode (no display window)")

# --- Import Config with Safety Check ---
try:
    import config
    # Verify it's the correct config file by checking for a known attribute
    if not hasattr(config, 'MODE'):
        print("[ERROR] Loaded the wrong 'config' module!")
        print(f"Path loaded: {config.__file__}")
        print("Please rename your local 'config.py' to 'flight_config.py' to avoid conflicts.")
        sys.exit(1)
except ImportError:
    print("[ERROR] Could not import 'config.py'. Make sure it exists in the same folder.")
    sys.exit(1)

from states import State
from utils import GeoTransformer
from planning import PathPlanner
from vision import VisionSystem

# Conditional Import for Simulation
if config.MODE == "SIMULATION":
    from simulation import SimulationEnvironment

class VisualFlightMission:
    def __init__(self):
        print(f"--- INITIALIZING IN {config.MODE} MODE ---")
        
        # 1. Initialize Geo & Map Tools
        if config.MODE == "SIMULATION":
            self.sim = SimulationEnvironment(GeoTransformer(map_w_px=100)) # Temp init
            self.geo = GeoTransformer(map_w_px=self.sim.map_w)
            self.sim.geo = self.geo # Sync geo tool
            # Returns only Target and Search Poly
            self.target_px, self.tgt_type, self.search_poly = self.sim.setup_on_map()
            
            self.eyes = VisionSystem(camera_index=None, model_path="best.tflite")
            if self.tgt_type == "dummy": self.eyes.using_ai = True
            else: self.eyes.using_ai = False
            
        else: # REAL MODE
            self.geo = GeoTransformer(map_w_px=4800) 
            self.sim = None
            self.search_poly = [] 
            
            self.eyes = VisionSystem(camera_index=config.REAL_CAMERA_INDEX, model_path="best.tflite")
            self.eyes.using_ai = True 
            print("Vision System: Real Camera Initialized")

        # 2. Planner
        self.planner = PathPlanner(self.geo, self.search_poly)

        # 3. Connection
        self.master = None
        self.last_req = 0
        self.last_heartbeat = 0
        
        # 4. State & Telemetry
        self.state = State.INIT
        self.previous_state = State.HOVER
        self.lat = config.REF_LAT
        self.lon = config.REF_LON
        self.alt = 0.0
        self.vx = 0; self.vy = 0; self.vz = 0
        self.roll = 0; self.pitch = 0; self.yaw = 0
        
        # 5. Mission Data
        self.waypoints = []
        self.wp_index = 0
        
        # Target Data
        self.target_lat = 0; self.target_lon = 0
        self.landing_lat = 0; self.landing_lon = 0
        self.current_conf = 0.0
        self.final_dist = 0.0 # Store final accuracy distance
        
        # Helper vars
        self.view_w_px = 100
        self.view_h_px = 100
        self.zoom_level = 1.0 
        self.last_speed_req = 0
        self.waiting_for_confirmation = False
        self.selecting_landing_side = False 
        
        # Logging
        self.log_file = open(config.LOG_FILE, 'w', newline='')
        self.logger = csv.writer(self.log_file)
        self.logger.writerow(["Timestamp", "State", "Lat", "Lon", "Alt", "Target_Conf"])

    def update_telemetry(self):
        if not self.master: return
        while True:
            msg = self.master.recv_match(blocking=False)
            if not msg: break
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.alt = msg.relative_alt / 1000.0
                self.vx = msg.vx / 100.0; self.vy = msg.vy / 100.0; self.vz = msg.vz / 100.0
            elif msg.get_type() == 'ATTITUDE':
                self.roll = msg.roll; self.pitch = msg.pitch; self.yaw = msg.yaw
            elif msg.get_type() == 'HEARTBEAT':
                self.last_heartbeat = time.time()

    def calculate_target_gps(self, u, v):
        Cx = config.IMAGE_W / 2; Cy = config.IMAGE_H / 2
        gsd_m = (config.SENSOR_WIDTH_MM * self.alt) / (config.FOCAL_LENGTH_MM * config.IMAGE_W)
        delta_x_px = u - Cx; delta_y_px = v - Cy 
        
        fwd_m = -delta_y_px * gsd_m
        right_m = delta_x_px * gsd_m
        offset_n = fwd_m * math.cos(self.yaw) - right_m * math.sin(self.yaw)
        offset_e = fwd_m * math.sin(self.yaw) + right_m * math.cos(self.yaw)
        
        R_EARTH = 6378137.0
        dLat = (offset_n / R_EARTH) * (180 / math.pi)
        dLon = (offset_e / (R_EARTH * math.cos(math.radians(self.lat)))) * (180 / math.pi)
        self.target_lat = self.lat + dLat
        self.target_lon = self.lon + dLon
        print(f"[VISION] Target GPS: {self.target_lat:.6f}, {self.target_lon:.6f}")

    def update_dashboard(self):
        found = False; u = 0; v = 0; conf = 0.0
        frame = None

        # 1. GET IMAGE FRAME
        if config.MODE == "SIMULATION":
            px, py = self.geo.gps_to_pixels(self.lat, self.lon)
            frame, self.view_w_px, self.view_h_px = self.sim.get_drone_view(px, py, self.alt, self.yaw)
        else:
            frame = self.eyes.get_frame()
            if frame is None: 
                frame = np.zeros((config.IMAGE_H, config.IMAGE_W, 3), dtype=np.uint8)

        # 2. PROCESS FRAME
        found, u, v, conf = self.eyes.process_frame_manually(frame)
        self.current_conf = conf

        # 3. DRAW HUD
        cx, cy = config.IMAGE_W // 2, config.IMAGE_H // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
        
        if found:
            cv2.circle(frame, (u, v), 15, (0, 255, 0), 2)
            cv2.line(frame, (u, v), (cx, cy), (0, 255, 0), 2)
            cv2.putText(frame, f"TGT {conf:.2f}", (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # --- INFO PANEL ---
        # Basic Stats
        cv2.putText(frame, f"MODE: {config.MODE}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"STATE: {self.state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"ALT: {self.alt:.1f}m", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Current Position (Always displayed)
        cv2.putText(frame, f"POS: {self.lat:.6f}, {self.lon:.6f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Target Location (If found/narrowing down)
        if self.target_lat != 0:
            cv2.putText(frame, f"TARGET: {self.target_lat:.6f}, {self.target_lon:.6f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Landing Location (If determined)
        if self.landing_lat != 0:
            cv2.putText(frame, f"LANDING: {self.landing_lat:.6f}, {self.landing_lon:.6f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Final Result (After Landing)
        if self.state == State.DONE:
             cv2.putText(frame, f"FINAL ERROR: {self.final_dist:.2f} m", (cx - 150, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        
        if self.waiting_for_confirmation:
             if self.selecting_landing_side:
                 cv2.putText(frame, "SELECT LANDING SIDE:", (cx - 200, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                 cv2.putText(frame, "N: North | S: South", (cx - 200, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                 cv2.putText(frame, "W: West  | E: East", (cx - 200, cy + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
             else:
                 cv2.putText(frame, "CONFIRM TARGET? (Y/N)", (cx - 150, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 4. COMPOSITE VIEW
        final_display = frame
        if config.MODE == "SIMULATION":
             god_frame = self.sim.get_god_view(
                px, py, self.yaw, self.view_w_px, self.view_h_px, self.zoom_level,
                self.planner.virtual_polygon, self.search_poly,
                (self.target_lat, self.target_lon), (self.landing_lat, self.landing_lon), self.geo
            )
             h_scale = frame.shape[0] / god_frame.shape[0]
             god_resized = cv2.resize(god_frame, (int(god_frame.shape[1]*h_scale), frame.shape[0]))
             final_display = np.hstack((god_resized, frame))

        if not HEADLESS:
            cv2.imshow("Mission Dashboard", final_display)
        return found, u, v

    def set_speed(self, speed_mps):
        if time.time() - self.last_speed_req < 3.0: return
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0, 1, speed_mps, -1, 0, 0, 0, 0)
        self.last_speed_req = time.time()

    def calculate_landing_spot(self, direction_key):
        R_EARTH = 6378137.0
        offset_dist = 7.5 # meters
        dLat = 0
        dLon = 0
        
        if direction_key == 'n': # North
            dLat = (offset_dist / R_EARTH) * (180/math.pi)
        elif direction_key == 's': # South
            dLat = -(offset_dist / R_EARTH) * (180/math.pi)
        elif direction_key == 'e': # East
            dLon = (offset_dist / (R_EARTH * math.cos(math.radians(self.target_lat)))) * (180/math.pi)
        elif direction_key == 'w': # West
            dLon = -(offset_dist / (R_EARTH * math.cos(math.radians(self.target_lat)))) * (180/math.pi)
            
        self.landing_lat = self.target_lat + dLat
        self.landing_lon = self.target_lon + dLon
        print(f"Landing Spot Selected: {direction_key.upper()} of Target")

    def run(self):
        print("Starting Mission Loop...")
        if not HEADLESS:
            cv2.namedWindow("Mission Dashboard")
            if config.MODE == "SIMULATION":
                cv2.setMouseCallback("Mission Dashboard", self.on_dashboard_mouse)

        key = -1
        last_status_print = 0
        while True:
            self.update_telemetry()
            target_found, px_u, px_v = self.update_dashboard()

            # Headless: print status to terminal periodically
            if HEADLESS and time.time() - last_status_print > 2.0:
                print(f"[{self.state}] Alt:{self.alt:.1f}m Pos:{self.lat:.6f},{self.lon:.6f} Conf:{self.current_conf:.2f}")
                last_status_print = time.time()
            
            # Log Data
            if time.time() % 1.0 < 0.1: 
                self.logger.writerow([datetime.now(), self.state, self.lat, self.lon, self.alt, self.current_conf])
            
            # --- KEY INPUTS ---
            if key == ord('m') or key == ord('M'):
                if self.state != State.MANUAL:
                    print("!!! MANUAL CONTROL OVERRIDE !!!")
                    self.previous_state = self.state 
                    self.state = State.MANUAL
                else:
                    print("Resuming Automation...")
                    if target_found: self.state = State.CENTERING
                    else: self.state = self.previous_state
            
            if self.state == State.VERIFY:
                if self.selecting_landing_side:
                    if key in [ord('n'), ord('e'), ord('w'), ord('s'), ord('N'), ord('E'), ord('W'), ord('S')]:
                        self.calculate_landing_spot(chr(key).lower())
                        self.selecting_landing_side = False
                        self.waiting_for_confirmation = False
                        self.state = State.APPROACH
                else:
                    if key == ord('y') or key == ord('Y'):
                        print("USER CONFIRMED TARGET. SELECT LANDING SIDE (N/E/W/S).")
                        self.selecting_landing_side = True
                    elif key == ord('n') or key == ord('N'):
                        print("USER REJECTED TARGET. RESUMING SEARCH.")
                        self.waiting_for_confirmation = False
                        self.state = State.SEARCH

            # --- STATE MACHINE ---
            if self.state == State.INIT:
                if time.time() - self.last_req > 1.0:
                    try:
                        print(f"Connecting to {config.CONNECTION_STR}...")
                        self.master = mavutil.mavlink_connection(config.CONNECTION_STR)
                        self.state = State.CONNECTING
                    except Exception as e: print(f"Connection fail: {e}")
                    self.last_req = time.time()

            elif self.state == State.CONNECTING:
                if self.last_heartbeat > 0:
                    print("Heartbeat. Requesting Data Stream...")
                    self.master.mav.request_data_stream_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
                    self.state = State.ARMING

            elif self.state == State.ARMING:
                if self.master.motors_armed():
                    print("Armed! Taking Off...")
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, config.TARGET_ALT)
                    self.state = State.TAKEOFF
                elif time.time() - self.last_req > 2.0:
                    self.master.mav.set_mode_send(
                        self.master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 
                        self.master.mode_mapping()['GUIDED'])
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()

            elif self.state == State.TAKEOFF:
                if self.alt >= config.TARGET_ALT * 0.90:
                    print("Target Altitude Reached.")
                    
                    if config.MODE == "SIMULATION":
                        self.waypoints = self.planner.generate_search_pattern(self.sim.map_w, self.sim.map_h, (self.lat, self.lon))
                    else:
                        print("Real Mode: No Map Polygon. Hovering or waiting for command...")
                        self.state = State.SEARCH 
                        
                    if self.waypoints:
                        # NEW LOGIC: Go to TRANSIT first, then SEARCH
                        print(f"Path generated. Transiting to start point: {self.waypoints[0]}")
                        self.state = State.TRANSIT_TO_SEARCH
                        self.last_speed_req = 0 
                    else:
                        print("No Waypoints generated.")
                        self.state = State.HOVER

            elif self.state == State.TRANSIT_TO_SEARCH:
                # Fly to the first waypoint of the search grid (Optimal Entry Point)
                self.set_speed(config.TRANSIT_SPEED_MPS)
                
                target = self.waypoints[0]
                if time.time() - self.last_req > 2.0:
                    self.send_global_target(target[0], target[1], config.TARGET_ALT)
                    self.last_req = time.time()
                
                # Check arrival
                if self.get_dist_to_point(target[0], target[1]) < 2.0:
                    print("Reached Search Start Point. Beginning Pattern.")
                    self.state = State.SEARCH
                    self.wp_index = 0 # Start from index 0

            elif self.state == State.SEARCH:
                self.set_speed(config.SEARCH_SPEED_MPS)
                if target_found:
                    print("TARGET DETECTED!")
                    self.calculate_target_gps(px_u, px_v)
                    self.state = State.CENTERING
                elif self.wp_index < len(self.waypoints):
                    target = self.waypoints[self.wp_index]
                    if time.time() - self.last_req > 2.0:
                        self.send_global_target(target[0], target[1], config.TARGET_ALT)
                        self.last_req = time.time()
                    if self.get_dist_to_point(target[0], target[1]) < 2.0:
                        self.wp_index += 1
                else:
                    self.state = State.DONE

            elif self.state == State.CENTERING:
                 if target_found: self.calculate_target_gps(px_u, px_v)
                 if time.time() - self.last_req > 0.2:
                     self.send_global_target(self.target_lat, self.target_lon, config.TARGET_ALT)
                     self.last_req = time.time()
                 if self.get_dist_to_target() < 1.0:
                     self.state = State.DESCENDING

            elif self.state == State.DESCENDING:
                 if target_found: self.calculate_target_gps(px_u, px_v)
                 if time.time() - self.last_req > 0.5:
                     self.send_global_target(self.target_lat, self.target_lon, config.VERIFY_ALT)
                     self.last_req = time.time()
                 if self.alt <= config.VERIFY_ALT + 1.0:
                     self.state = State.VERIFY

            elif self.state == State.VERIFY:
                self.waiting_for_confirmation = True
                self.send_global_target(self.target_lat, self.target_lon, config.VERIFY_ALT)

            elif self.state == State.APPROACH:
                self.send_global_target(self.landing_lat, self.landing_lon, config.VERIFY_ALT)
                if self.get_dist_to_point(self.landing_lat, self.landing_lon) < 1.0:
                    self.state = State.LANDING

            elif self.state == State.LANDING:
                if self.alt < 0.3:
                    print("Touchdown. Disarming.")
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)
                    
                    # Calculate final error
                    lat_scale = 111132.0 
                    final_error = math.sqrt(((self.lat-self.target_lat)*lat_scale)**2 + ((self.lon-self.target_lon)*lat_scale*0.62)**2)
                    self.final_dist = final_error
                    print(f"MISSION COMPLETE. Final Error: {self.final_dist:.2f} m")
                    self.state = State.DONE
                else:
                    self.send_global_target(self.landing_lat, self.landing_lon, 0) 

            if not HEADLESS:
                key = cv2.waitKey(20) & 0xFF
                if key == 27: break
            else:
                time.sleep(0.02)

    def on_dashboard_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0: self.zoom_level = min(self.zoom_level * 1.2, 20.0)
            else: self.zoom_level = max(self.zoom_level / 1.2, 1.0)
            
    def send_global_target(self, lat, lon, alt):
        self.master.mav.set_position_target_global_int_send(
             0, self.master.target_system, self.master.target_component,
             mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
             0b110111111000, int(lat * 1e7), int(lon * 1e7), alt, 0, 0, 0, 0, 0, 0, 0, 0)
             
    def get_dist_to_target(self): return self.get_dist_to_point(self.target_lat, self.target_lon)
    def get_dist_to_point(self, t_lat, t_lon):
        lat_scale = 111132.0 
        return math.sqrt(((self.lat-t_lat)*lat_scale)**2 + ((self.lon-t_lon)*lat_scale*0.62)**2)

if __name__ == "__main__":
    mission = VisualFlightMission()
    mission.run()