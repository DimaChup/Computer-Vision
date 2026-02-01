from pymavlink import mavutil
import time
import math
import sys
import cv2
import numpy as np
import heapq # For pathfinding
from vision import VisionSystem

# ==========================================
#       CALIBRATION SECTION
# ==========================================
CONNECTION_STR = 'tcp:127.0.0.1:5762'
TARGET_ALT = 40.0 # Search Altitude
VERIFY_ALT = 15.0 # Descent Altitude for Verification

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

# 4. Safety
NFZ_BUFFER_M = 10.0 # Meters to keep away from NFZ

# 5. Speed Settings (NEW)
TRANSIT_SPEED_MPS = 15.0  # Speed when flying TO the search area (m/s)
SEARCH_SPEED_MPS = 10.0   # Speed while scanning the grid (m/s)
# ==========================================

# --- STATES ---
class State:
    INIT = "INIT"
    CONNECTING = "CONNECTING"
    ARMING = "ARMING"
    TAKEOFF = "TAKEOFF"
    TRANSIT_TO_SEARCH = "TRANSIT_TO_SEARCH"
    SEARCH = "SEARCH"
    CENTERING = "CENTERING" 
    DESCENDING = "DESCENDING"
    VERIFY = "VERIFY"
    HOVER = "HOVER"         
    APPROACH = "APPROACH"   
    LANDING = "LANDING"
    MANUAL = "MANUAL"
    SAFETY_HALT = "SAFETY_HALT" 
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
        
        # 2. Initialize Coverage Overlay
        self.coverage_overlay = np.zeros_like(self.full_map)
        
        # Load Dummy Asset
        self.dummy_img = cv2.imread(DUMMY_FILE, cv2.IMREAD_UNCHANGED)
        if self.dummy_img is None:
            print(f"Warning: '{DUMMY_FILE}' not found. Stickman mode only.")
        
        # --- CALCULATE SIZES ---
        # 1. Red Dot Size (on Map)
        raw_radius = TARGET_REAL_RADIUS_M * self.pix_per_m
        self.target_radius_px = max(2, int(raw_radius))
        
        print(f"Map Scale: 1m = {self.pix_per_m:.2f} px")
        
        # 3. Interactive Setup (Target + Search Area + NFZ)
        self.sim_target_type = None
        self.sim_target_px = None
        self.search_polygon = [] 
        self.nfz_polygon = [] 
        self.manual_transit_pixels = [] # NEW: Store manual waypoints here
        self.search_polygon_gps = []
        self.virtual_polygon = [] # The inset area (Purple)
        
        self.setup_simulation_on_map()
        
        # 4. Vision System
        self.eyes = VisionSystem(camera_index=None, model_path="best.tflite")
        # Configure vision mode based on what user placed
        if self.sim_target_type == "dummy":
            self.eyes.using_ai = True 
        else:
            self.eyes.using_ai = False 
        
        # 5. Flight State
        self.state = State.INIT
        self.previous_state = State.HOVER
        self.master = None
        self.last_req = 0
        self.last_heartbeat = 0
        
        # 6. Telemetry
        self.lat = REF_LAT
        self.lon = REF_LON
        self.alt = 0.0
        self.vx = 0.0 # NEW: X Velocity
        self.vy = 0.0 # NEW: Y Velocity
        self.vz = 0.0
        self.roll = 0; self.pitch = 0; self.yaw = 0
        self.current_conf = 0.0
        
        # 7. Mission Data
        self.waypoints = []
        self.wp_index = 0
        self.transit_waypoints = [] # Smart path to search area
        self.transit_index = 0
        
        self.target_lat = 0
        self.target_lon = 0
        self.landing_lat = 0
        self.landing_lon = 0
        self.final_dist = 0.0
        
        # View Params
        self.view_w_px = 100
        self.view_h_px = 100
        self.zoom_level = 1.0 
        
        # Control Flags
        self.last_speed_req = 0 # Timer for robust speed setting

    def overlay_image_alpha(self, background, overlay, x, y, target_w, target_h, rotation_deg=0):
        if overlay is None: return
        h_src, w_src = overlay.shape[:2]
        if target_w == 0:
            scale = target_h / h_src
            target_w = int(w_src * scale)
        if target_w <= 0 or target_h <= 0: return
        try: resized = cv2.resize(overlay, (target_w, target_h))
        except: return 
        
        diag = int(math.sqrt(target_w**2 + target_h**2))
        pad_x = (diag - target_w) // 2
        pad_y = (diag - target_h) // 2
        padded = cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0,0,0,0))
        h_pad, w_pad = padded.shape[:2]
        M_rot = cv2.getRotationMatrix2D((w_pad//2, h_pad//2), rotation_deg, 1.0)
        rotated = cv2.warpAffine(padded, M_rot, (w_pad, h_pad))
        
        y1 = y - h_pad // 2; y2 = y1 + h_pad
        x1 = x - w_pad // 2; x2 = x1 + w_pad
        h_bg, w_bg = background.shape[:2]
        y1_c = max(0, y1); y2_c = min(h_bg, y2)
        x1_c = max(0, x1); x2_c = min(w_bg, x2)
        if y1_c >= y2_c or x1_c >= x2_c: return
        ov_y1 = y1_c - y1; ov_y2 = ov_y1 + (y2_c - y1_c)
        ov_x1 = x1_c - x1; ov_x2 = ov_x1 + (x2_c - x1_c)
        overlay_crop = rotated[ov_y1:ov_y2, ov_x1:ov_x2]
        bg_crop = background[y1_c:y2_c, x1_c:x2_c]
        if overlay_crop.shape[2] == 4:
            alpha = overlay_crop[:, :, 3] / 255.0
            for c in range(0, 3):
                bg_crop[:, :, c] = (1. - alpha) * bg_crop[:, :, c] + alpha * overlay_crop[:, :, c]
        else:
            background[y1_c:y2_c, x1_c:x2_c] = overlay_crop

    def setup_simulation_on_map(self):
        # Resize for display
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
        self.search_polygon = [] 
        self.polygon_closed = False
        
        self.nfz_polygon = []
        self.nfz_closed = False
        self.transit_done = False # NEW: Flag for manual waypoint phase
        
        self.window_name = "Select Target" 

        def mouse_callback(event, x, y, flags, param):
            real_x = int(x / scale_factor)
            real_y = int(y / scale_factor)
            
            update = False

            if event == cv2.EVENT_LBUTTONDOWN:
                # PHASE 1: PLACE TARGET (If not set)
                if self.sim_target_px is None:
                    self.sim_target_px = (real_x, real_y)
                    # Visuals
                    if flags & cv2.EVENT_FLAG_CTRLKEY:
                        self.temp_type = "dummy"
                    else:
                        self.temp_type = "dot"
                    print("Step 1 Done: Target Placed. Now draw SEARCH Polygon.")
                
                # PHASE 2: SEARCH POLYGON
                elif not self.polygon_closed:
                    self.search_polygon.append((real_x, real_y))
                
                # PHASE 3: NFZ POLYGON
                elif not self.nfz_closed:
                      self.nfz_polygon.append((real_x, real_y))
                
                # NEW PHASE 4: MANUAL TRANSIT POINTS
                elif not self.transit_done:
                      self.manual_transit_pixels.append((real_x, real_y))
                
                update = True

            elif event == cv2.EVENT_RBUTTONDOWN:
                if not self.polygon_closed and len(self.search_polygon) >= 3:
                    self.polygon_closed = True
                    print("Step 2 Done: Search Poly Closed. Now draw NO-FLY ZONE (Red).")
                    update = True
                elif self.polygon_closed and not self.nfz_closed:
                    # Allow closing NFZ even if points < 3 to skip
                    self.nfz_closed = True
                    print("Step 3 Done: NFZ Closed. Click INTERMEDIATE WAYPOINTS (Cyan) or R-Click to Launch.")
                    update = True
                # NEW: Close Transit Phase
                elif self.nfz_closed and not self.transit_done:
                    self.transit_done = True
                    print("Step 4 Done: Transit Setup Complete. Press KEY to Launch.")
                    update = True

            if update:
                temp_vis = display_map.copy()
                # Target
                if self.sim_target_px:
                      sx, sy = int(self.sim_target_px[0]*scale_factor), int(self.sim_target_px[1]*scale_factor)
                      if self.temp_type == "dummy":
                          map_h_px = DUMMY_HEIGHT_M * self.pix_per_m
                          display_h_px = max(20, int(map_h_px * scale_factor))
                          self.overlay_image_alpha(temp_vis, self.dummy_img, sx, sy, 0, display_h_px)
                      else:
                          vis_radius = max(2, int(self.target_radius_px * scale_factor))
                          cv2.circle(temp_vis, (sx, sy), vis_radius, (0, 0, 255), -1)

                # Search Poly (Green)
                if len(self.search_polygon) > 0:
                    pts = [np.array([[int(p[0]*scale_factor), int(p[1]*scale_factor)] for p in self.search_polygon], dtype=np.int32)]
                    cv2.polylines(temp_vis, pts, self.polygon_closed, (0, 255, 0), 2)
                    for p in pts[0]: cv2.circle(temp_vis, tuple(p), 3, (0, 255, 0), -1)
                
                # NFZ Poly (Red)
                if len(self.nfz_polygon) > 0:
                    pts = [np.array([[int(p[0]*scale_factor), int(p[1]*scale_factor)] for p in self.nfz_polygon], dtype=np.int32)]
                    cv2.polylines(temp_vis, pts, self.nfz_closed, (0, 0, 255), 2)
                    for p in pts[0]: cv2.circle(temp_vis, tuple(p), 3, (0, 0, 255), -1)

                # NEW: Manual Transit Points (Cyan)
                if len(self.manual_transit_pixels) > 0:
                    t_pts = [np.array([[int(p[0]*scale_factor), int(p[1]*scale_factor)] for p in self.manual_transit_pixels], dtype=np.int32)]
                    cv2.polylines(temp_vis, t_pts, False, (255, 255, 0), 1)
                    for p in t_pts[0]: cv2.circle(temp_vis, tuple(p), 4, (255, 255, 0), -1)

                cv2.imshow(self.window_name, temp_vis)

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, display_map)
        cv2.setMouseCallback(self.window_name, mouse_callback)
        print("--- MISSION SETUP ---")
        print("1. Click Target.")
        print("2. Left-Click Search Points -> Right-Click Close (Green).")
        print("3. Left-Click NFZ Points -> Right-Click Close (Red).")
        print("4. Left-Click Intermediate Waypoints -> Right-Click Finish (Cyan).")
        print("5. Press KEY to start.")
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)
        
        # Save Final State
        if hasattr(self, 'sim_target_px'):
            self.sim_target_type = self.temp_type

        # Convert Polygon Pixels to GPS for Flight Logic
        for pt in self.search_polygon:
            lat, lon = self.pixels_to_gps(pt[0], pt[1])
            self.search_polygon_gps.append((lat, lon))

    def gps_to_pixels(self, lat, lon):
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
        lon_m = 111132.954 * math.cos(math.radians(lat))
        dy = -(lat - REF_LAT) * lat_m
        dx = (lon - REF_LON) * lon_m
        return int(dx * self.pix_per_m), int(dy * self.pix_per_m)
        
    def pixels_to_gps(self, x, y):
        dx = x / self.pix_per_m
        dy = y / self.pix_per_m
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(REF_LAT))
        lon_m = 111132.954 * math.cos(math.radians(REF_LAT))
        dLat = -(dy / lat_m)
        dLon = dx / lon_m
        return REF_LAT + dLat, REF_LON + dLon

    def check_geofence_violation(self):
        """ Returns (True/False, distance_to_edge_m) """
        if len(self.nfz_polygon) < 3: return False, 999.0
        
        # Current Drone Position in Pixels
        dx, dy = self.gps_to_pixels(self.lat, self.lon)
        
        # Poly for CV2
        pts = np.array([self.nfz_polygon], dtype=np.int32)
        
        # pointPolygonTest: +ve if inside, -ve if outside
        # We trigger if inside OR within buffer distance
        dist_px = cv2.pointPolygonTest(pts, (dx, dy), True)
        dist_m = dist_px / self.pix_per_m
        
        # If dist_m > -10.0, we are closer than 10m to the NFZ (or inside it)
        # Note: 10m is the configured NFZ_BUFFER_M
        if dist_m > -NFZ_BUFFER_M:
            return True, dist_m
            
        return False, dist_m

    def get_drone_view(self, cx, cy):
        # 1. Setup Camera Params
        fov = 2 * math.atan(SENSOR_WIDTH_MM / (2 * FOCAL_LENGTH_MM))
        safe_alt = max(1.0, self.alt)
        ground_w = 2 * safe_alt * math.tan(fov / 2)
        
        self.view_w_px = int(ground_w * self.pix_per_m)
        self.view_h_px = int(self.view_w_px * (IMAGE_H / IMAGE_W))
        
        # 2. Optimized Crop-Then-Rotate Logic
        diag = int(math.sqrt(self.view_w_px**2 + self.view_h_px**2))
        x1 = cx - diag // 2; y1 = cy - diag // 2
        x2 = x1 + diag; y2 = y1 + diag
        
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
            
        center = (diag // 2, diag // 2)
        M = cv2.getRotationMatrix2D(center, math.degrees(self.yaw), 1.0)
        rotated_patch = cv2.warpAffine(raw_crop, M, (diag, diag))
        
        start_x = (diag - self.view_w_px) // 2
        start_y = (diag - self.view_h_px) // 2
        crop = rotated_patch[start_y:start_y+self.view_h_px, start_x:start_x+self.view_w_px]
        
        final_view = cv2.resize(crop, (IMAGE_W, IMAGE_H))
            
        if self.sim_target_px is not None:
            dx = self.sim_target_px[0] - cx
            dy = self.sim_target_px[1] - cy
            angle_rad = -self.yaw
            dx_rot = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            dy_rot = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            scale = IMAGE_W / max(1, self.view_w_px)
            screen_x = int((IMAGE_W / 2) + (dx_rot * scale))
            screen_y = int((IMAGE_H / 2) + (dy_rot * scale))
            px_per_m_screen = IMAGE_W / ground_w
            
            if self.sim_target_type == "dummy" and self.dummy_img is not None:
                dummy_h_screen = int(DUMMY_HEIGHT_M * px_per_m_screen)
                self.overlay_image_alpha(final_view, self.dummy_img, screen_x, screen_y, 0, dummy_h_screen, rotation_deg=math.degrees(self.yaw))
            else:
                dot_rad_screen = int(TARGET_REAL_RADIUS_M * px_per_m_screen)
                cv2.circle(final_view, (screen_x, screen_y), max(3, dot_rad_screen), (0, 0, 255), -1)

        return final_view

    def get_god_view(self, cx, cy):
        display_map = self.full_map.copy()
        
        if self.sim_target_px is not None:
            if self.sim_target_type == "dummy" and self.dummy_img is not None:
                map_h_px = int(DUMMY_HEIGHT_M * self.pix_per_m)
                map_h_px = max(10, map_h_px) 
                self.overlay_image_alpha(display_map, self.dummy_img, self.sim_target_px[0], self.sim_target_px[1], 0, map_h_px)
            else:
                cv2.circle(display_map, self.sim_target_px, self.target_radius_px, (0, 0, 255), -1)

        # Draw Polygons
        if len(self.search_polygon) > 1:
              cv2.polylines(display_map, [np.array(self.search_polygon, np.int32)], True, (0, 255, 0), 2)
        if len(self.nfz_polygon) > 1:
              cv2.polylines(display_map, [np.array(self.nfz_polygon, np.int32)], True, (0, 0, 255), 2)
        
        # Draw Virtual Area (Purple) - Showing the "Edge of FOV" flight boundary
        if len(self.virtual_polygon) > 0:
              # We might have contours (list of points), convert to format for polylines
              cv2.drawContours(display_map, [self.virtual_polygon], -1, (255, 0, 255), 2)

        # Draw Coverage (Correctly blended)
        rect = ((cx, cy), (self.view_w_px, self.view_h_px), math.degrees(self.yaw))
        box = np.int32(cv2.boxPoints(rect))
        cv2.fillPoly(self.coverage_overlay, [box], (255, 255, 0)) 
        cv2.addWeighted(self.coverage_overlay, 0.2, display_map, 1.0, 0, display_map)
        
        # Draw Drone
        cv2.circle(display_map, (cx, cy), 8, (255, 0, 0), -1)
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
        
        # FIX: UNPACK 4 VALUES (Updated VisionSystem)
        found, u, v, conf = self.eyes.process_frame_manually(drone_frame)
        self.current_conf = conf

        if found:
            cv2.circle(drone_frame, (u, v), 10, (0, 255, 0), 2)
            cv2.line(drone_frame, (u, v), (cx, cy), (0, 255, 0), 1) 
            cv2.putText(drone_frame, f"TARGET ({conf*100:.0f}%)", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Dashboard Text
        cv2.putText(drone_frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(drone_frame, f"Alt: {self.alt:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # NEW: Show Dynamic FOV Size
        fov_w_m = (SENSOR_WIDTH_MM * max(1.0, self.alt)) / FOCAL_LENGTH_MM
        fov_h_m = fov_w_m * (IMAGE_H / IMAGE_W)
        cv2.putText(drone_frame, f"FOV: {fov_w_m:.1f}m x {fov_h_m:.1f}m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # NEW: Show Ground Speed
        g_speed = math.sqrt(self.vx**2 + self.vy**2)
        cv2.putText(drone_frame, f"Speed: {g_speed:.1f} m/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.state == State.MANUAL:
            cv2.putText(drone_frame, "MANUAL CONTROL: W/S/A/D/Q/E I/K", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.state == State.SAFETY_HALT:
            cv2.putText(drone_frame, "!!! SAFETY HALT !!!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(drone_frame, "NFZ BREACH DETECTED", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(drone_frame, "Press 'M' to take Manual Control", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if self.target_lat != 0:
            cv2.putText(drone_frame, f"Tgt GPS: {self.target_lat:.6f}, {self.target_lon:.6f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.landing_lat != 0:
            cv2.putText(drone_frame, f"Land GPS: {self.landing_lat:.6f}, {self.landing_lon:.6f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if self.state == State.DONE:
              cv2.putText(drone_frame, f"FINAL LANDING: {self.final_dist:.2f}m from Target", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.state == State.SEARCH:
              cv2.putText(drone_frame, f"WP: {self.wp_index}/{len(self.waypoints)}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if god_frame.shape[0] != drone_frame.shape[0]:
            god_frame = cv2.resize(god_frame, (int(god_frame.shape[1] * (drone_frame.shape[0]/god_frame.shape[0])), drone_frame.shape[0]))

        combined = np.hstack((god_frame, drone_frame))
        # SCALE FOR DISPLAY
        if combined.shape[1] > 1280:
            scale = 1280 / combined.shape[1]
            combined = cv2.resize(combined, (0,0), fx=scale, fy=scale)
            
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
                self.vx = msg.vx / 100.0 # North/X (m/s)
                self.vy = msg.vy / 100.0 # East/Y (m/s)
                self.vz = msg.vz / 100.0 # Down/Z (m/s)
            elif msg.get_type() == 'ATTITUDE':
                self.roll = msg.roll
                self.pitch = msg.pitch
                self.yaw = msg.yaw
            elif msg.get_type() == 'HEARTBEAT':
                self.last_heartbeat = time.time()

    def plan_path_around_nfz(self, start_gps, end_gps):
        if len(self.nfz_polygon) < 3: return [end_gps]
            
        start_px = self.gps_to_pixels(start_gps[0], start_gps[1])
        end_px = self.gps_to_pixels(end_gps[0], end_gps[1])
        
        def intersect(A, B, C, D):
            def ccw(p1, p2, p3):
                return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        def line_blocked(p1, p2, poly):
            for i in range(len(poly)):
                p3 = poly[i]
                p4 = poly[(i+1)%len(poly)]
                if intersect(p1, p2, p3, p4): return True
            return False

        if not line_blocked(start_px, end_px, self.nfz_polygon):
            return [end_gps] 
            
        print("[PATH] Direct path blocked by NFZ. Computing avoidance...")

        nodes = [start_px, end_px]
        poly_centroid = np.mean(self.nfz_polygon, axis=0)
        buffer_factor = 1.2
        for pt in self.nfz_polygon:
            vec = np.array(pt) - poly_centroid
            new_pt = poly_centroid + vec * buffer_factor
            nodes.append(tuple(new_pt.astype(int)))
            
        adj = {i: [] for i in range(len(nodes))}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not line_blocked(nodes[i], nodes[j], self.nfz_polygon):
                    dist = math.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
                    adj[i].append((dist, j))
                    adj[j].append((dist, i))

        # --- LOGIC UPDATE: SQUARE OFF CORNERS ---
        # If a node sees the target, but a neighbor is closer (or roughly equal) and also sees it, 
        # remove the direct link to force perimeter following.
        corner_start_idx = 2
        num_corners = len(nodes) - 2
        
        # Tolerance: Allow neighbor to be slightly further (15m) to prefer squaring off
        tolerance_px = 45.0 * self.pix_per_m 
        
        if num_corners > 1:
            # Helper to find distance in adjacency list
            def get_edge_dist(u, target_node):
                for d, v in adj[u]:
                    if v == target_node: return d
                return None

            for i in range(num_corners):
                u = corner_start_idx + i
                
                # If u connects to End (Node 1)
                dist_u_end = get_edge_dist(u, 1)
                if dist_u_end is not None:
                    # Check Polygon Neighbors
                    # Neighbors in 'nodes' list are sequential: 2, 3, 4...
                    # We need to wrap around the polygon indices
                    poly_idx = i
                    next_poly = (poly_idx + 1) % num_corners
                    prev_poly = (poly_idx - 1 + num_corners) % num_corners
                    
                    neighbors = [corner_start_idx + next_poly, corner_start_idx + prev_poly]
                    
                    for v in neighbors:
                        dist_v_end = get_edge_dist(v, 1)
                        dist_u_v = get_edge_dist(u, v)
                        
                        # If neighbor V connects to End AND connects to U
                        if dist_v_end is not None and dist_u_v is not None:
                            # AGGRESSIVE RAIL RIDING:
                            # Prefer the neighbor path (u->v->Target) even if v is slightly further
                            # from target than u (within tolerance). This forces the drone to "corner" 
                            # properly instead of cutting diagonally as soon as it clears the obstacle.
                            if dist_v_end < (dist_u_end + tolerance_px):
                                # Cut the direct path from U -> 1, forcing U -> V -> 1
                                adj[u] = [edge for edge in adj[u] if edge[1] != 1]
                                break

        pq = [(0, 0, [])] 
        visited = set()
        
        while pq:
            cost, u, path = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            
            path = path + [u]
            if u == 1: 
                gps_path = []
                for node_idx in path[1:]: 
                    px = nodes[node_idx]
                    lat, lon = self.pixels_to_gps(px[0], px[1])
                    gps_path.append((lat, lon))
                return gps_path

            for weight, v in adj[u]:
                if v not in visited:
                    heapq.heappush(pq, (cost + weight, v, path))
                    
        print("[PATH] No path found!")
        return [end_gps]

    def generate_search_pattern(self):
        print("Generating OPTIMIZED Search Grid with NFZ Avoidance & Inset...")
        if len(self.search_polygon) < 3:
            print("WARNING: No polygon defined. Using default relative box.")
            return 

        # 1. Create Initial Mask
        mask = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
        poly_pts = np.array([self.search_polygon], dtype=np.int32)
        cv2.fillPoly(mask, poly_pts, 255)
        
        # --- NEW: CALCULATE INSET ---
        ground_width_m = (SENSOR_WIDTH_MM * TARGET_ALT) / FOCAL_LENGTH_MM
        inset_dist_m = 13.0 # Fixed to 5.0m as requested
        inset_px = int(inset_dist_m * self.pix_per_m)
        
        if inset_px > 0:
            print(f"Applying Search Inset of {inset_dist_m:.1f}m ({inset_px}px) for FOV alignment.")
            
            # Erode to create the "Virtual Area" (Safe Zone)
            kernel_size = 2 * inset_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            eroded_mask = cv2.erode(mask, kernel)
            
            # Extract and store for visualization
            contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Find largest contour to ensure we get the main area
                largest_contour = max(contours, key=cv2.contourArea)
                self.virtual_polygon = largest_contour # Save largest for drawing later
                mask = eroded_mask # Apply erosion
            else:
                print("Warning: Search area too small for FOV inset. Using full area.")
                self.virtual_polygon = []
        
        # Subtract NFZ if it exists
        if len(self.nfz_polygon) >= 3:
            nfz_mask = np.zeros_like(mask)
            nfz_pts = np.array([self.nfz_polygon], dtype=np.int32)
            cv2.fillPoly(nfz_mask, nfz_pts, 255)
            # Subtract NFZ from search mask
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(nfz_mask))
            
            # Re-extract contour of the safe area (after NFZ subtraction) for correct perimeter flying
            safe_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if safe_contours:
                 # Update virtual polygon to match the safe, flyable area
                 self.virtual_polygon = max(safe_contours, key=cv2.contourArea)

        # 2. Determine Optimal Sweep Direction
        rect = cv2.minAreaRect(poly_pts[0])
        (center, size, angle) = rect
        width, height = size
        scan_angle = angle + 90 if width < height else angle
        
        print(f"Optimal Scan Angle: {scan_angle:.1f} degrees")

        # 3. Rotate Mask
        M = cv2.getRotationMatrix2D(center, scan_angle, 1.0)
        M_inv = cv2.invertAffineTransform(M)
        rotated_mask = cv2.warpAffine(mask, M, (self.map_w, self.map_h))

        # 4. Generate Scan Lines
        overlap = 0.2
        SWATH_M = ground_width_m * (1.0 - overlap)
        step_px = int(SWATH_M * self.pix_per_m)
        if step_px < 1: step_px = 1
        
        points = cv2.findNonZero(rotated_mask)
        if points is None: return 
        x, y, w, h = cv2.boundingRect(points)
        
        perimeter_wps = []
        wps = []
        direction = 1 

        # 5. Generate Perimeter Waypoints (Furthest Edge First)
        if len(self.virtual_polygon) > 0:
            print("Adding Perimeter Waypoints...")
            # Approximate contour to reduce points
            epsilon = 0.005 * cv2.arcLength(self.virtual_polygon, True)
            approx = cv2.approxPolyDP(self.virtual_polygon, epsilon, True)
            
            # Convert to list of points
            cnt_pts = approx.reshape(-1, 2)
            
            # Find point FURTHEST from drone current location
            curr_px = self.gps_to_pixels(self.lat, self.lon)
            dists = np.sum((cnt_pts - curr_px)**2, axis=1)
            furthest_idx = np.argmax(dists) # Changed from argmin to argmax
            
            # Reorder points to start from furthest
            ordered_pts = np.roll(cnt_pts, -furthest_idx, axis=0)
            
            # Add to list
            for pt in ordered_pts:
                lat, lon = self.pixels_to_gps(pt[0], pt[1])
                perimeter_wps.append((lat, lon))
            
            # Close the loop
            if len(perimeter_wps) > 0:
                perimeter_wps.append(perimeter_wps[0])

        
        # 6. Generate Sweep Lines
        for scan_y in range(y + step_px//2, y + h, step_px):
            row = rotated_mask[scan_y, :]
            pixels = np.where(row == 255)[0]
            
            if len(pixels) > 0:
                diffs = np.diff(pixels)
                breaks = np.where(diffs > 1)[0]
                segment_starts = [0] + list(breaks + 1)
                segment_ends = list(breaks) + [len(pixels)-1]
                
                segments = []
                for i in range(len(segment_starts)):
                    x_start = pixels[segment_starts[i]]
                    x_end = pixels[segment_ends[i]]
                    segments.append((x_start, x_end))
                
                if direction == -1:
                    segments.reverse()

                for x_start, x_end in segments:
                    pt1 = np.array([[[x_start, scan_y]]], dtype=np.float32)
                    pt2 = np.array([[[x_end, scan_y]]], dtype=np.float32)
                    pt1_orig = cv2.transform(pt1, M_inv)[0][0]
                    pt2_orig = cv2.transform(pt2, M_inv)[0][0]
                    
                    p1_gps = self.pixels_to_gps(pt1_orig[0], pt1_orig[1])
                    p2_gps = self.pixels_to_gps(pt2_orig[0], pt2_orig[1])
                    
                    if direction == 1:
                        wps.extend([((p1_gps[0], p1_gps[1])), ((p2_gps[0], p2_gps[1]))])
                    else:
                        wps.extend([((p2_gps[0], p2_gps[1])), ((p1_gps[0], p1_gps[1]))])
                
                direction *= -1

        self.waypoints = perimeter_wps + wps
        print(f"Generated {len(self.waypoints)} waypoints (Perimeter + Sweep).")

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

    # NEW: Helper function to change drone speed
    def set_speed(self, speed_mps):
        # Prevent spamming: only send if > 3 seconds since last send
        if time.time() - self.last_speed_req < 3.0: return
        
        print(f"[CMD] Setting Ground Speed to {speed_mps} m/s")
        # 1 = Ground Speed type
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            1, speed_mps, -1, 0, 0, 0, 0
        )
        self.last_speed_req = time.time()

    def run(self):
        print("Starting Mission...")
        cv2.namedWindow("Mission Dashboard")
        cv2.setMouseCallback("Mission Dashboard", self.on_dashboard_mouse)
        key = -1 
        while True:
            self.update_telemetry()
            target_found, px_u, px_v = self.update_dashboard()
            
            # --- KEY INPUTS (PROCESS FIRST) ---
            if key == ord('m') or key == ord('M'):
                if self.state != State.MANUAL:
                    print(f"[{self.state}] SWITCHING TO MANUAL CONTROL.")
                    if self.state != State.SAFETY_HALT:
                        self.previous_state = self.state 
                    self.state = State.MANUAL
                else:
                    print(f"[{self.state}] EXITING MANUAL.")
                    # SMART RESUME: Check if target visible
                    if target_found:
                         print("Target in view! Switching to CENTERING sequence.")
                         self.calculate_target_gps(px_u, px_v)
                         self.state = State.CENTERING
                    else:
                         print("Resuming previous mission state.")
                         self.state = self.previous_state if self.previous_state else State.HOVER
                    
                    self.master.mav.set_position_target_local_ned_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                        0b110111000111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                key = -1
            
            if key == ord('r') or key == ord('R'):
                if self.master.motors_armed():
                    print("MANUAL CMD: DISARMING")
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)
                else:
                    print("MANUAL CMD: ARMING")
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                key = -1

            # --- SAFETY WATCHDOG ---
            violation, dist = self.check_geofence_violation()
            
            # CRITICAL FIX: Only trigger Halt if NOT in Manual
            if violation and self.state != State.MANUAL:
                if self.state != State.SAFETY_HALT:
                    print(f"[SAFETY] NFZ PROXIMITY ({dist:.1f}m). AUTOMATIC BRAKE.")
                    self.previous_state = self.state # Remember where we were
                    self.state = State.SAFETY_HALT
            
            # --- FLIGHT STATE MACHINE ---
            if self.state == State.SAFETY_HALT:
                 # Force Stop/Brake
                 self.master.mav.set_position_target_local_ned_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                        0b110111000111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                 # Allow manual override to escape
                 if key == ord('m') or key == ord('M'):
                      self.state = State.MANUAL

            elif self.state == State.INIT:
                if time.time() - self.last_req > 1.0:
                    try:
                        self.master = mavutil.mavlink_connection(CONNECTION_STR)
                        self.state = State.CONNECTING
                    except: pass
                    self.last_req = time.time()

            elif self.state == State.CONNECTING:
                if self.last_heartbeat > 0:
                    self.master.mav.request_data_stream_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
                    )
                    self.state = State.ARMING
                else:
                    if time.time() - self.last_req > 1.0:
                        self.last_req = time.time()

            elif self.state == State.ARMING:
                if time.time() - self.last_req > 2.0:
                    mode_id = self.master.mode_mapping()['GUIDED']
                    self.master.mav.set_mode_send(
                        self.master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()

                if self.master.motors_armed():
                    self.state = State.TAKEOFF
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, TARGET_ALT)

            elif self.state == State.TAKEOFF:
                if self.alt >= TARGET_ALT * 0.90:
                    self.generate_search_pattern()
                    # --- NEW LOGIC: CHECK FOR MANUAL TRANSIT POINTS FIRST ---
                    if len(self.manual_transit_pixels) > 0:
                        print(f"Loading {len(self.manual_transit_pixels)} Manual Transit Points...")
                        self.transit_waypoints = []
                        # Convert all manual pixels to GPS
                        for px in self.manual_transit_pixels:
                            lat, lon = self.pixels_to_gps(px[0], px[1])
                            self.transit_waypoints.append((lat, lon))
                        
                        # IMPORTANT: Append the first waypoint of the Search Grid so we fly INTO the grid
                        if len(self.waypoints) > 0:
                            self.transit_waypoints.append(self.waypoints[0])
                            
                        self.transit_index = 0
                        self.state = State.TRANSIT_TO_SEARCH
                        self.last_speed_req = 0 # Force immediate send on transition
                    
                    elif self.waypoints:
                        # FALLBACK: PLAN AUTO TRANSIT PATH if no manual points
                        self.transit_waypoints = self.plan_path_around_nfz(
                            (self.lat, self.lon), self.waypoints[0])
                        self.transit_index = 0
                        self.state = State.TRANSIT_TO_SEARCH
                        self.last_speed_req = 0 # Force immediate send on transition
                    else:
                        self.state = State.SEARCH
                        self.last_speed_req = 0 # Force immediate send on transition

                elif time.time() - self.last_req > 5.0:
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, TARGET_ALT)
                    self.last_req = time.time()
            
            elif self.state == State.TRANSIT_TO_SEARCH:
                # --- ROBUST SPEED SETTING ---
                self.set_speed(TRANSIT_SPEED_MPS)

                if self.transit_index < len(self.transit_waypoints):
                    target = self.transit_waypoints[self.transit_index]
                    lat_scale = 111132.0 
                    dist = math.sqrt(((self.lat-target[0])*lat_scale)**2 + ((self.lon-target[1])*lat_scale*0.62)**2)
                    
                    if dist < 2.0:
                        self.transit_index += 1
                        print(f"Reached Transit Waypoint {self.transit_index}")
                    elif time.time() - self.last_req > 2.0:
                        self.master.mav.set_position_target_global_int_send(
                            0, self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                            0b110111111000, 
                            int(target[0] * 1e7), int(target[1] * 1e7), TARGET_ALT,
                            0, 0, 0, 0, 0, 0, 0, 0)
                        self.last_req = time.time()
                else:
                     print(f"[{self.state}] Arrived at Search Grid.")
                     self.state = State.SEARCH
                     self.last_speed_req = 0 # Force immediate send on transition

            elif self.state == State.SEARCH:
                # --- ROBUST SPEED SETTING ---
                self.set_speed(SEARCH_SPEED_MPS)

                if target_found:
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
                elif key == 82 or key == ord('i'): vz = -1.0 
                elif key == 84 or key == ord('k'): vz = 1.0  
                
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
                    self.state = State.DESCENDING
                    self.last_req = time.time()

            elif self.state == State.DESCENDING:
                if target_found:
                     self.calculate_target_gps(px_u, px_v)
                
                if time.time() - self.last_req > 0.5:
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.target_lat * 1e7), int(self.target_lon * 1e7), VERIFY_ALT, 
                        0, 0, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()
                
                if self.alt <= (VERIFY_ALT + 1.0): 
                      self.state = State.VERIFY
                      self.last_req = time.time() 

            elif self.state == State.VERIFY:
                 if time.time() - self.last_req > 2.0:
                     if self.current_conf > 0.75:
                          R_EARTH = 6378137.0
                          offset_dist = 7.5
                          offset_deg_lat = (offset_dist / R_EARTH) * (180 / math.pi)
                          self.landing_lat = self.target_lat + offset_deg_lat
                          self.landing_lon = self.target_lon
                          self.state = State.APPROACH
                     else:
                          self.master.mav.set_position_target_global_int_send(
                            0, self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                            0b110111111000, 
                            int(self.target_lat * 1e7), int(self.target_lon * 1e7), VERIFY_ALT,
                            0, 0, 0, 0, 0, 0, 0, 0)

            elif self.state == State.APPROACH:
                if time.time() - self.last_req > 2.0:
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.landing_lat * 1e7), int(self.landing_lon * 1e7), VERIFY_ALT, 
                        0, 0, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()
                
                lat_scale = 111132.0 
                dist = math.sqrt(((self.lat-self.landing_lat)*lat_scale)**2 + ((self.lon-self.landing_lon)*lat_scale*0.62)**2)
                
                if dist < 1.0:
                    self.state = State.LANDING

            elif self.state == State.LANDING:
                is_landed = False
                if self.alt < 0.3: is_landed = True
                if self.alt < 10.0 and abs(self.vz) < 0.1 and time.time() - self.last_req > 5.0: is_landed = True

                if is_landed:
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)
                    
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