# filename: test_flight.py
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
TARGET_ALT = 35.0 # Meters

# 1. Map Configuration
MAP_FILE = "map.jpg"
DUMMY_FILE = "dummy.png"
MAP_WIDTH_METERS = 480.0  # Total width of the map image in meters
REF_LAT = 51.425106  
REF_LON = -2.672257

# 2. Target Size Configuration
TARGET_REAL_RADIUS_M = 0.15 # 15 cm radius for the red dot
DUMMY_HEIGHT_M = 1.8        # 6 ft (1.8m) tall dummy

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
    MANUAL = "MANUAL"
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
        
        # Load Dummy Asset
        self.dummy_img = cv2.imread(DUMMY_FILE, cv2.IMREAD_UNCHANGED)
        if self.dummy_img is None:
            print(f"Warning: '{DUMMY_FILE}' not found. Stickman mode only.")
        
        # --- CALCULATE SIZES ---
        # 1. Red Dot Size (on Map)
        raw_radius = TARGET_REAL_RADIUS_M * self.pix_per_m
        self.target_radius_px = max(2, int(raw_radius))
        
        print(f"Map Scale: 1m = {self.pix_per_m:.2f} px")
        
        # 2. Interactive Target Placement
        self.sim_target_type = None
        self.sim_target_px = None
        
        self.select_target_on_map()
        
        # 3. Vision System
        # We assume best.tflite is in the folder
        self.eyes = VisionSystem(camera_index=None, model_path="best.tflite")
        # Configure vision mode based on what user placed
        if self.sim_target_type == "dummy":
            self.eyes.using_ai = True # Force AI check if available
        else:
            self.eyes.using_ai = False # Force Color if Dot
        
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
        self.zoom_level = 1.0 

    def overlay_image_alpha(self, background, overlay, x, y, target_w, target_h, rotation_deg=0):
        if overlay is None: return
        
        h_src, w_src = overlay.shape[:2]
        if target_w == 0:
            scale = target_h / h_src
            target_w = int(w_src * scale)
            
        if target_w <= 0 or target_h <= 0: return

        try:
            resized = cv2.resize(overlay, (target_w, target_h))
        except: return 
        
        # Rotation with padding to prevent corner clipping
        diag = int(math.sqrt(target_w**2 + target_h**2))
        pad_x = (diag - target_w) // 2
        pad_y = (diag - target_h) // 2
        
        padded = cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0,0,0,0))
        h_pad, w_pad = padded.shape[:2]
        
        M_rot = cv2.getRotationMatrix2D((w_pad//2, h_pad//2), rotation_deg, 1.0)
        rotated = cv2.warpAffine(padded, M_rot, (w_pad, h_pad))
        
        # Calculate bounds on background
        y1 = y - h_pad // 2; y2 = y1 + h_pad
        x1 = x - w_pad // 2; x2 = x1 + w_pad
        
        h_bg, w_bg = background.shape[:2]
        
        # Clipping logic
        y1_c = max(0, y1); y2_c = min(h_bg, y2)
        x1_c = max(0, x1); x2_c = min(w_bg, x2)
        
        if y1_c >= y2_c or x1_c >= x2_c: return
        
        ov_y1 = y1_c - y1; ov_y2 = ov_y1 + (y2_c - y1_c)
        ov_x1 = x1_c - x1; ov_x2 = ov_x1 + (x2_c - x1_c)
        
        overlay_crop = rotated[ov_y1:ov_y2, ov_x1:ov_x2]
        bg_crop = background[y1_c:y2_c, x1_c:x2_c]
        
        # Alpha blending
        if overlay_crop.shape[2] == 4:
            alpha = overlay_crop[:, :, 3] / 255.0
            for c in range(0, 3):
                bg_crop[:, :, c] = (1. - alpha) * bg_crop[:, :, c] + alpha * overlay_crop[:, :, c]
        else:
            background[y1_c:y2_c, x1_c:x2_c] = overlay_crop

    def select_target_on_map(self):
        # Resize for display if map is too big
        h, w = self.full_map.shape[:2]
        MAX_DISPLAY_H = 800  
        
        scale_factor = 1.0
        if h > MAX_DISPLAY_H:
            scale_factor = MAX_DISPLAY_H / h
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            display_map = cv2.resize(self.full_map, (new_w, new_h))
        else:
            display_map = self.full_map.copy()

        self.temp_type = "dot"

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                temp_vis = display_map.copy()
                
                real_x = int(x / scale_factor)
                real_y = int(y / scale_factor)
                self.sim_target_px = (real_x, real_y)
                
                # Check Control Key for Dummy
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    self.temp_type = "dummy"
                    # Visualize dummy scaled for this preview window
                    map_h_px = DUMMY_HEIGHT_M * self.pix_per_m
                    display_h_px = int(map_h_px * scale_factor)
                    display_h_px = max(20, display_h_px) 
                    
                    self.overlay_image_alpha(temp_vis, self.dummy_img, x, y, 0, display_h_px)
                else:
                    self.temp_type = "dot"
                    vis_radius = max(2, int(self.target_radius_px * scale_factor))
                    cv2.circle(temp_vis, (x, y), vis_radius, (0, 0, 255), -1)
                
                cv2.imshow("Select Target", temp_vis)

        cv2.imshow("Select Target", display_map)
        cv2.setMouseCallback("Select Target", mouse_callback)
        print("CLICK map. Left: Red Dot. Ctrl+Left: Dummy. Press KEY to start.")
        cv2.waitKey(0)
        cv2.destroyWindow("Select Target")
        
        if hasattr(self, 'sim_target_px'):
            self.sim_target_type = self.temp_type
            # Note: We do NOT burn it into self.full_map. We render dynamically.

    def gps_to_pixels(self, lat, lon):
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
        lon_m = 111132.954 * math.cos(math.radians(lat))
        
        dy = -(lat - REF_LAT) * lat_m
        dx = (lon - REF_LON) * lon_m
        
        return int(dx * self.pix_per_m), int(dy * self.pix_per_m)

    def get_drone_view(self, cx, cy):
        # 1. Setup Camera Params
        fov = 2 * math.atan(SENSOR_WIDTH_MM / (2 * FOCAL_LENGTH_MM))
        safe_alt = max(1.0, self.alt)
        ground_w = 2 * safe_alt * math.tan(fov / 2)
        
        self.view_w_px = int(ground_w * self.pix_per_m)
        self.view_h_px = int(self.view_w_px * (IMAGE_H / IMAGE_W))
        
        # 2. Optimized Crop-Then-Rotate Logic
        diag = int(math.sqrt(self.view_w_px**2 + self.view_h_px**2))
        
        x1 = cx - diag // 2
        y1 = cy - diag // 2
        x2 = x1 + diag
        y2 = y1 + diag
        
        # Handle Map Boundaries with padding
        pad_l = max(0, -x1); pad_t = max(0, -y1)
        pad_r = max(0, x2 - self.map_w); pad_b = max(0, y2 - self.map_h)
        
        sx1 = x1 + pad_l; sy1 = y1 + pad_t
        sx2 = x2 - pad_r; sy2 = y2 - pad_b
        
        if sx2 > sx1 and sy2 > sy1:
            raw_crop = self.full_map[sy1:sy2, sx1:sx2]
            if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
                raw_crop = cv2.copyMakeBorder(raw_crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
        else:
            raw_crop = np.zeros((diag, diag, 3), dtype=np.uint8)
            
        # Rotate the small crop
        center = (diag // 2, diag // 2)
        M = cv2.getRotationMatrix2D(center, math.degrees(self.yaw), 1.0)
        rotated_patch = cv2.warpAffine(raw_crop, M, (diag, diag))
        
        # Extract camera view
        start_x = (diag - self.view_w_px) // 2
        start_y = (diag - self.view_h_px) // 2
        crop = rotated_patch[start_y:start_y+self.view_h_px, start_x:start_x+self.view_w_px]
        
        # 3. Resize to Screen Resolution (Blurry Grass)
        final_view = cv2.resize(crop, (IMAGE_W, IMAGE_H))
            
        # 4. Dynamic Target Overlay (Sharp Object)
        if self.sim_target_px is not None:
            # Calculate position relative to drone
            dx = self.sim_target_px[0] - cx
            dy = self.sim_target_px[1] - cy
            
            # Rotate by -yaw to align with camera frame
            angle_rad = -self.yaw
            dx_rot = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            dy_rot = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            
            # Convert to Screen Pixels
            scale = IMAGE_W / max(1, self.view_w_px)
            screen_x = int((IMAGE_W / 2) + (dx_rot * scale))
            screen_y = int((IMAGE_H / 2) + (dy_rot * scale))
            
            # Calculate object size on screen
            px_per_m_screen = IMAGE_W / ground_w
            
            # Render
            if self.sim_target_type == "dummy" and self.dummy_img is not None:
                dummy_h_screen = int(DUMMY_HEIGHT_M * px_per_m_screen)
                # Rotate +yaw so dummy stays fixed on ground
                self.overlay_image_alpha(final_view, self.dummy_img, screen_x, screen_y, 0, dummy_h_screen, rotation_deg=math.degrees(self.yaw))
            else:
                dot_rad_screen = int(TARGET_REAL_RADIUS_M * px_per_m_screen)
                cv2.circle(final_view, (screen_x, screen_y), max(3, dot_rad_screen), (0, 0, 255), -1)

        return final_view

    def get_god_view(self, cx, cy):
        display_map = self.full_map.copy()
        
        # Draw Target (Marker)
        if self.sim_target_px is not None:
            if self.sim_target_type == "dummy" and self.dummy_img is not None:
                map_h_px = int(DUMMY_HEIGHT_M * self.pix_per_m)
                map_h_px = max(10, map_h_px) 
                self.overlay_image_alpha(display_map, self.dummy_img, self.sim_target_px[0], self.sim_target_px[1], 0, map_h_px)
            else:
                cv2.circle(display_map, self.sim_target_px, self.target_radius_px, (0, 0, 255), -1)

        # Draw Drone
        cv2.circle(display_map, (cx, cy), 8, (255, 0, 0), -1)
        rect = ((cx, cy), (self.view_w_px, self.view_h_px), math.degrees(self.yaw))
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(display_map, [box], 0, (0, 255, 255), 2)
        
        if self.target_lat != 0:
            tx, ty = self.gps_to_pixels(self.target_lat, self.target_lon)
            cv2.circle(display_map, (tx, ty), 5, (0, 255, 0), -1) 
        
        if self.landing_lat != 0:
            lx, ly = self.gps_to_pixels(self.landing_lat, self.landing_lon)
            cv2.circle(display_map, (lx, ly), 5, (255, 0, 255), -1) 
            cv2.circle(display_map, (lx, ly), 20, (255, 255, 255), 1)

        # Apply Zoom
        if self.zoom_level > 1.0:
            h, w = display_map.shape[:2]
            crop_h = int(h / self.zoom_level)
            crop_w = int(w / self.zoom_level)
            x1 = max(0, min(w - crop_w, cx - crop_w // 2))
            y1 = max(0, min(h - crop_h, cy - crop_h // 2))
            x2 = x1 + crop_w; y2 = y1 + crop_h
            display_map = display_map[y1:y2, x1:x2]

        target_h = IMAGE_H
        base_scale = target_h / self.map_h
        target_w = int(self.map_w * base_scale)
        if self.zoom_level > 1.0: return cv2.resize(display_map, (IMAGE_H, IMAGE_H))
        return cv2.resize(display_map, (target_w, target_h))

    def on_dashboard_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0: self.zoom_level = min(self.zoom_level * 1.2, 20.0)
            else: self.zoom_level = max(self.zoom_level / 1.2, 1.0)

    def update_dashboard(self):
        px, py = self.gps_to_pixels(self.lat, self.lon)
        
        drone_frame = self.get_drone_view(px, py)
        god_frame = self.get_god_view(px, py)
        
        cx, cy = IMAGE_W // 2, IMAGE_H // 2
        cv2.line(drone_frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 0), 2)
        cv2.line(drone_frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 0), 2)
        
        found, u, v = self.eyes.process_frame_manually(drone_frame)
        if found:
            # Note: VisionSystem draws box if AI detects, we draw circle for Dot fallback
            cv2.circle(drone_frame, (u, v), 10, (0, 255, 0), 2)
            cv2.line(drone_frame, (u, v), (cx, cy), (0, 255, 0), 1) 
            cv2.putText(drone_frame, "TARGET ACQUIRED", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Dashboard Text
        cv2.putText(drone_frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(drone_frame, f"Alt: {self.alt:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.state == State.MANUAL:
            cv2.putText(drone_frame, "MANUAL CONTROL: W/S/A/D/Q/E", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if self.target_lat != 0:
            cv2.putText(drone_frame, f"Tgt GPS: {self.target_lat:.6f}, {self.target_lon:.6f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.landing_lat != 0:
            cv2.putText(drone_frame, f"Land GPS: {self.landing_lat:.6f}, {self.landing_lon:.6f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if self.state == State.DONE:
             cv2.putText(drone_frame, f"FINAL LANDING: {self.final_dist:.2f}m from Target", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Ensure height match for hstack
        if god_frame.shape[0] != drone_frame.shape[0]:
            god_frame = cv2.resize(god_frame, (int(god_frame.shape[1] * (drone_frame.shape[0]/god_frame.shape[0])), drone_frame.shape[0]))

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
        cv2.namedWindow("Mission Dashboard")
        cv2.setMouseCallback("Mission Dashboard", self.on_dashboard_mouse)
        key = -1 
        while True:
            self.update_telemetry()
            target_found, px_u, px_v = self.update_dashboard()
            
            if key == ord('m') or key == ord('M'):
                if self.state != State.MANUAL:
                    print(f"[{self.state}] SWITCHING TO MANUAL CONTROL.")
                    self.state = State.MANUAL
                else:
                    print(f"[{self.state}] EXITING MANUAL. HOVERING.")
                    self.state = State.HOVER
                    self.master.mav.set_position_target_local_ned_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                        0b110111000111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                key = -1

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
            
            elif self.state == State.MANUAL:
                vx = 0; vy = 0; vz = 0; yaw_rate = 0
                SPEED = 3.0 
                if key == ord('w'): vx = SPEED     
                elif key == ord('s'): vx = -SPEED  
                elif key == ord('a'): vy = -SPEED  
                elif key == ord('d'): vy = SPEED   
                elif key == ord('q'): yaw_rate = -0.5 
                elif key == ord('e'): yaw_rate = 0.5  
                elif key == 82: vz = -1.0 
                elif key == 84: vz = 1.0  
                type_mask = 0b010111000111 
                self.master.mav.set_position_target_local_ned_send(
                    0, self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                    type_mask, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)

            elif self.state == State.CENTERING:
                if target_found:
                    self.calculate_target_gps(px_u, px_v)
                if time.time() - self.last_req > 0.5: 
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.target_lat * 1e7), int(self.target_lon * 1e7), TARGET_ALT,
                        0, 0, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()
                lat_scale = 111132.0
                dist = math.sqrt(((self.lat-self.target_lat)*lat_scale)**2 + ((self.lon-self.target_lon)*lat_scale*0.62)**2)
                if dist < 0.5:
                    print(f"[{self.state}] ALIGNED (Error: {dist:.2f}m). Hovering for confidence.")
                    self.state = State.HOVER
                    self.last_req = time.time()

            elif self.state == State.HOVER:
                if time.time() - self.last_req > 5.0:
                    print(f"[{self.state}] Target Verified. Calculating Landing Offset.")
                    R_EARTH = 6378137.0
                    offset_dist = 7.5
                    offset_deg_lat = (offset_dist / R_EARTH) * (180 / math.pi)
                    self.landing_lat = self.target_lat + offset_deg_lat
                    self.landing_lon = self.target_lon
                    print(f"Landing Target: {self.landing_lat:.6f}, {self.landing_lon:.6f}")
                    self.state = State.APPROACH
                    self.last_req = time.time()
                else:
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
                    lat_scale = 111132.0
                    final_error = math.sqrt(((self.lat-self.target_lat)*lat_scale)**2 + ((self.lon-self.target_lon)*lat_scale*0.62)**2)
                    self.final_dist = final_error
                    self.state = State.DONE
                elif time.time() - self.last_req > 1.0:
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.landing_lat * 1e7), int(self.landing_lon * 1e7), 0, 
                        0, 0, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()

            key = cv2.waitKey(20) & 0xFF
            if key == 27: break
            if key == ord('+') or key == ord('='): self.zoom_level = min(self.zoom_level * 1.2, 20.0)
            if key == ord('-') or key == ord('_'): self.zoom_level = max(self.zoom_level / 1.2, 1.0)

if __name__ == "__main__":
    mission = VisualFlightMission()
    mission.run()