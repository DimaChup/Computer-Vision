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
TARGET_ALT = 50.0 # Meters

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
    TRANSIT_TO_SEARCH = "TRANSIT_TO_SEARCH"
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
        
        # 3. Interactive Setup (Target + Search Area)
        self.sim_target_type = None
        self.sim_target_px = None
        self.search_polygon = [] # List of (x,y) pixels
        
        self.setup_simulation_on_map()
        
        # 4. Vision System
        self.eyes = VisionSystem(camera_index=None, model_path="best.tflite")
        # Configure vision mode based on what user placed
        if self.sim_target_type == "dummy":
            self.eyes.using_ai = True # Force AI check if available
        else:
            self.eyes.using_ai = False # Force Color if Dot
        
        # 5. Flight State
        self.state = State.INIT
        self.master = None
        self.last_req = 0
        self.last_heartbeat = 0
        
        # 6. Telemetry
        self.lat = REF_LAT
        self.lon = REF_LON
        self.alt = 0.0
        self.roll = 0; self.pitch = 0; self.yaw = 0
        
        # 7. Mission Data
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
                    print("Step 1 Complete: Target Placed. Now Click points for Search Polygon.")
                
                # PHASE 2: DEFINE SEARCH POLYGON POINTS
                elif not self.polygon_closed:
                    self.search_polygon.append((real_x, real_y))
                    print(f"Added Polygon Point {len(self.search_polygon)}")
                
                update = True

            elif event == cv2.EVENT_RBUTTONDOWN:
                # PHASE 3: CLOSE POLYGON
                if len(self.search_polygon) >= 3:
                    self.polygon_closed = True
                    print("Polygon Closed. Press KEY to Launch.")
                    update = True

            if update:
                temp_vis = display_map.copy()
                
                # Draw Target
                if self.sim_target_px:
                     sx, sy = int(self.sim_target_px[0]*scale_factor), int(self.sim_target_px[1]*scale_factor)
                     if self.temp_type == "dummy":
                         map_h_px = DUMMY_HEIGHT_M * self.pix_per_m
                         display_h_px = max(20, int(map_h_px * scale_factor))
                         self.overlay_image_alpha(temp_vis, self.dummy_img, sx, sy, 0, display_h_px)
                     else:
                         vis_radius = max(2, int(self.target_radius_px * scale_factor))
                         cv2.circle(temp_vis, (sx, sy), vis_radius, (0, 0, 255), -1)

                # Draw Polygon Points & Lines
                poly_pts = []
                for pt in self.search_polygon:
                    px, py = int(pt[0]*scale_factor), int(pt[1]*scale_factor)
                    cv2.circle(temp_vis, (px, py), 4, (0, 255, 255), -1)
                    poly_pts.append([px, py])
                
                if len(poly_pts) > 1:
                    cv2.polylines(temp_vis, [np.array(poly_pts)], self.polygon_closed, (0, 255, 255), 2)

                cv2.imshow(self.window_name, temp_vis)

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, display_map)
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        print("--- MISSION SETUP ---")
        print("1. Left Click to place Target (Ctrl+Click for Dummy).")
        print("2. Left Click multiple points to draw Search Polygon.")
        print("3. Right Click to CLOSE Polygon.")
        print("4. Press KEY to start.")
        
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)
        
        # Save Final State
        if hasattr(self, 'sim_target_px'):
            self.sim_target_type = self.temp_type

    def gps_to_pixels(self, lat, lon):
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
        lon_m = 111132.954 * math.cos(math.radians(lat))
        dy = -(lat - REF_LAT) * lat_m
        dx = (lon - REF_LON) * lon_m
        return int(dx * self.pix_per_m), int(dy * self.pix_per_m)
        
    def pixels_to_gps(self, x, y):
        # Convert pixels to delta meters
        dx = x / self.pix_per_m
        dy = y / self.pix_per_m
        # Meters to GPS
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(REF_LAT))
        lon_m = 111132.954 * math.cos(math.radians(REF_LAT))
        
        # dy is positive DOWN (image), which is negative Lat (South)
        dLat = -(dy / lat_m)
        dLon = dx / lon_m
        
        return REF_LAT + dLat, REF_LON + dLon

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
        
        # 3. Resize to Screen Resolution
        final_view = cv2.resize(crop, (IMAGE_W, IMAGE_H))
            
        # 4. Dynamic Target Overlay (Sharp Object)
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
        
        # Draw Target
        if self.sim_target_px is not None:
            if self.sim_target_type == "dummy" and self.dummy_img is not None:
                map_h_px = int(DUMMY_HEIGHT_M * self.pix_per_m)
                map_h_px = max(10, map_h_px) 
                self.overlay_image_alpha(display_map, self.dummy_img, self.sim_target_px[0], self.sim_target_px[1], 0, map_h_px)
            else:
                cv2.circle(display_map, self.sim_target_px, self.target_radius_px, (0, 0, 255), -1)

        # Draw Search Polygon
        if len(self.search_polygon) > 1:
             poly_pts = [np.array(self.search_polygon, np.int32)]
             cv2.polylines(display_map, poly_pts, True, (0, 255, 255), 2)

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
        
        found, u, v = self.eyes.process_frame_manually(drone_frame)
        if found:
            cv2.circle(drone_frame, (u, v), 10, (0, 255, 0), 2)
            cv2.line(drone_frame, (u, v), (cx, cy), (0, 255, 0), 1) 
            cv2.putText(drone_frame, "TARGET ACQUIRED", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Dashboard Text
        cv2.putText(drone_frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(drone_frame, f"Alt: {self.alt:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.state == State.MANUAL:
            cv2.putText(drone_frame, "MANUAL CONTROL: W/S/A/D/Q/E I/K", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if self.target_lat != 0:
            cv2.putText(drone_frame, f"Tgt GPS: {self.target_lat:.6f}, {self.target_lon:.6f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.landing_lat != 0:
            cv2.putText(drone_frame, f"Land GPS: {self.landing_lat:.6f}, {self.landing_lon:.6f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if self.state == State.DONE:
             cv2.putText(drone_frame, f"FINAL LANDING: {self.final_dist:.2f}m from Target", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.state == State.SEARCH:
             cv2.putText(drone_frame, f"WP: {self.wp_index}/{len(self.waypoints)}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if god_frame.shape[0] != drone_frame.shape[0]:
            god_frame = cv2.resize(god_frame, (int(god_frame.shape[1] * (drone_frame.shape[0]/god_frame.shape[0])), drone_frame.shape[0]))

        combined = np.hstack((god_frame, drone_frame))
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
            elif msg.get_type() == 'ATTITUDE':
                self.roll = msg.roll
                self.pitch = msg.pitch
                self.yaw = msg.yaw
            elif msg.get_type() == 'HEARTBEAT':
                self.last_heartbeat = time.time()

    def generate_search_pattern(self):
        print("Generating Search Grid from Polygon...")
        if len(self.search_polygon) < 3:
            print("WARNING: No polygon defined. Using default relative box.")
            width_m, height_m = 120.0, 120.0
            start_lat, start_lon = self.lat, self.lon
            # Create default box polygon for masking
            tl_px = self.gps_to_pixels(start_lat, start_lon)
            br_px = (int(tl_px[0] + width_m*self.pix_per_m), int(tl_px[1] + height_m*self.pix_per_m))
            poly_mask_pts = np.array([[tl_px, (br_px[0], tl_px[1]), br_px, (tl_px[0], br_px[1])]], dtype=np.int32)
        else:
            # Use the user-defined polygon
            poly_mask_pts = np.array([self.search_polygon], dtype=np.int32)
            
            # Find bounds of the polygon
            x_pts = [p[0] for p in self.search_polygon]
            y_pts = [p[1] for p in self.search_polygon]
            min_x, max_x = min(x_pts), max(x_pts)
            min_y, max_y = min(y_pts), max(y_pts)

        # Create Binary Mask of the Search Area
        mask = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
        cv2.fillPoly(mask, poly_mask_pts, 255)

        # --- DYNAMIC SWATH CALCULATION ---
        # Ground Width = (Sensor_Width * Altitude) / Focal_Length
        ground_width_m = (SENSOR_WIDTH_MM * TARGET_ALT) / FOCAL_LENGTH_MM
        overlap = 0.2 # 20% overlap
        SWATH_M = ground_width_m * (1.0 - overlap)
        print(f"Computed Swath: {SWATH_M:.2f}m (Width: {ground_width_m:.2f}m at {TARGET_ALT}m Alt)")
        
        step_px = int(SWATH_M * self.pix_per_m)
        if step_px < 1: step_px = 1
        
        # Scan bounds (from min_y to max_y)
        scan_y_start = min_y if 'min_y' in locals() else 0
        scan_y_end = max_y if 'max_y' in locals() else self.map_h
        
        wps = []
        direction = 1 # 1 = East, -1 = West
        
        for y in range(scan_y_start + step_px//2, scan_y_end, step_px):
            # Find intersection of scanline Y with polygon mask
            row = mask[y, :]
            pixels = np.where(row == 255)[0]
            
            if len(pixels) > 0:
                x_start = pixels[0]
                x_end = pixels[-1]
                
                # Convert to GPS
                lat_1, lon_1 = self.pixels_to_gps(x_start, y)
                lat_2, lon_2 = self.pixels_to_gps(x_end, y)
                
                if direction == 1:
                    wps.append((lat_1, lon_1))
                    wps.append((lat_2, lon_2))
                else:
                    wps.append((lat_2, lon_2))
                    wps.append((lat_1, lon_1))
                
                direction *= -1

        self.waypoints = wps
        print(f"Generated {len(wps)} waypoints.")

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
                    print(f"\n[{self.state}] Altitude Reached. Flying to Grid Start.")
                    
                    # New: Fly to start of grid if defined
                    self.generate_search_pattern()
                    if self.waypoints:
                        start_wp = self.waypoints[0]
                        self.master.mav.set_position_target_global_int_send(
                            0, self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                            0b110111111000, 
                            int(start_wp[0] * 1e7), int(start_wp[1] * 1e7), TARGET_ALT,
                            0, 0, 0, 0, 0, 0, 0, 0)
                        self.state = State.TRANSIT_TO_SEARCH
                    else:
                        self.state = State.SEARCH

                elif time.time() - self.last_req > 5.0:
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, TARGET_ALT)
                    self.last_req = time.time()
            
            elif self.state == State.TRANSIT_TO_SEARCH:
                if self.waypoints:
                    target = self.waypoints[0]
                    # Check distance to start point
                    lat_scale = 111132.0 
                    dist = math.sqrt(((self.lat-target[0])*lat_scale)**2 + ((self.lon-target[1])*lat_scale*0.62)**2)
                    
                    if dist < 2.0:
                        print(f"[{self.state}] Reached Grid Start. Beginning Search.")
                        self.state = State.SEARCH
                    elif time.time() - self.last_req > 2.0:
                        self.master.mav.set_position_target_global_int_send(
                            0, self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                            0b110111111000, 
                            int(target[0] * 1e7), int(target[1] * 1e7), TARGET_ALT,
                            0, 0, 0, 0, 0, 0, 0, 0)
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
                elif key == 82 or key == ord('i'): vz = -1.0 # Up Arrow (Climb)
                elif key == 84 or key == ord('k'): vz = 1.0  # Down Arrow (Descend)
                
                type_mask = 0b010111000111 
                self.master.mav.set_position_target_local_ned_send(
                    0, self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                    type_mask, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)

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

            # Keyboard Zoom support as backup
            key = cv2.waitKey(20) & 0xFF
            if key == 27: break
            if key == ord('+') or key == ord('='): self.zoom_level = min(self.zoom_level * 1.2, 20.0)
            if key == ord('-') or key == ord('_'): self.zoom_level = max(self.zoom_level / 1.2, 1.0)

if __name__ == "__main__":
    mission = VisualFlightMission()
    mission.run()