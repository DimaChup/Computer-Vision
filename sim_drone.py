# filename: sim_drone.py
import cv2
import numpy as np
import math
from vision import VisionSystem
from geolocation import pixel_to_gps

# --- SIM CONFIG ---
MAP_FILE = "map.jpg"      # Screenshot of the field
MAP_WIDTH_METERS = 200.0  # Width of that screenshot in meters
START_POS = (10, 10)      # Drone starts top-left

# Camera Specs (Same as real)
SENSOR_WIDTH_MM = 5.02; FOCAL_LENGTH_MM = 6.0
IMAGE_W = 640; IMAGE_H = 480

class DroneSim:
    def __init__(self):
        # 1. Load Map
        self.full_map = cv2.imread(MAP_FILE)
        if self.full_map is None:
            print(f"Map file '{MAP_FILE}' not found. Creating default green map.")
            self.full_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
            self.full_map[:] = (34, 139, 34) # Green Field
            cv2.line(self.full_map, (0, 0), (1000, 1000), (30, 100, 30), 5)
            
        self.map_h, self.map_w = self.full_map.shape[:2]
        self.pix_per_m = self.map_w / MAP_WIDTH_METERS
        
        # 2. Interactive Target Placement
        self.target_pos = (50, 50) # Default
        self.select_target_on_map()
        
        # 3. Draw Target (Red Dot) permanently on the base map
        tx = int(self.target_pos[0] * self.pix_per_m)
        ty = int(self.target_pos[1] * self.pix_per_m)
        cv2.circle(self.full_map, (tx, ty), 6, (0, 0, 255), -1)
        
        # 4. Sim State
        self.x = START_POS[0]; self.y = START_POS[1]
        self.alt = 35.0; self.yaw = 0.0
        self.mode = "AUTO" # AUTO or MANUAL
        self.target_found = False
        
        # 5. Generate Search Pattern (Lawnmower)
        self.waypoints = self.generate_lawnmower_pattern()
        self.current_wp_index = 0
        
        self.eyes = VisionSystem(camera_index=None)

    def select_target_on_map(self):
        """ Opens a window to let user click the target location """
        temp_map = self.full_map.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert pixel back to meters
                mx = x / self.pix_per_m
                my = y / self.pix_per_m
                self.target_pos = (mx, my)
                print(f"Target set to: {mx:.1f}m, {my:.1f}m")
                # Draw preview
                cv2.circle(temp_map, (x, y), 6, (0, 0, 255), -1)
                cv2.imshow("Select Target", temp_map)

        cv2.imshow("Select Target", temp_map)
        cv2.setMouseCallback("Select Target", mouse_callback)
        print("CLICK on map to place target. Press ANY KEY to start simulation.")
        cv2.waitKey(0)
        cv2.destroyWindow("Select Target")

    def generate_lawnmower_pattern(self):
        """ Generates zigzag waypoints covering the map """
        waypoints = []
        margin = 20.0 # meters from edge
        spacing = 30.0 # meters between legs (based on camera swath width)
        
        width_m = self.map_w / self.pix_per_m
        height_m = self.map_h / self.pix_per_m
        
        x_min = margin; x_max = width_m - margin
        y_current = margin
        going_right = True
        
        while y_current < height_m - margin:
            if going_right:
                waypoints.append((x_min, y_current))
                waypoints.append((x_max, y_current))
            else:
                waypoints.append((x_max, y_current))
                waypoints.append((x_min, y_current))
                
            y_current += spacing
            going_right = not going_right
            
        return waypoints

    def update_autopilot(self):
        """ Logic to fly towards the next waypoint """
        if self.target_found:
            return # Stop moving if found
            
        if self.current_wp_index >= len(self.waypoints):
            print("Search Complete. Hovering.")
            return

        target_x, target_y = self.waypoints[self.current_wp_index]
        
        # Vector to target
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Heading to target
        target_yaw = math.atan2(dy, dx) + (math.pi/2) # +90 deg offset for North=Up logic if needed, usually atan2 is fine
        # Sim uses 0=Right (East), pi/2=Down (South) for screen coords typically
        target_yaw = math.atan2(dy, dx) 

        # Turn towards target (Simple P controller)
        yaw_err = target_yaw - self.yaw
        # Normalize angle
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
        self.yaw += yaw_err * 0.1
        
        # Move forward
        speed = 8.0 * 0.1 # m/s per tick approx
        if dist > 2.0:
            self.x += math.cos(self.yaw) * speed
            self.y += math.sin(self.yaw) * speed
        else:
            # Waypoint reached
            self.current_wp_index += 1
            print(f"Waypoint {self.current_wp_index} Reached")

    def get_drone_view(self):
        """ Generates the Camera View (Right Side) """
        # Calculate FOV crop
        fov_rad = 2 * math.atan(SENSOR_WIDTH_MM / (2 * FOCAL_LENGTH_MM))
        ground_w = 2 * self.alt * math.tan(fov_rad / 2)
        
        # Convert ground dimensions to map pixels
        self.view_w_px = int(ground_w * self.pix_per_m)
        self.view_h_px = int(self.view_w_px * (IMAGE_H / IMAGE_W))
        
        # Crop & Rotate logic
        cx, cy = int(self.x * self.pix_per_m), int(self.y * self.pix_per_m)
        
        # Get Rotation Matrix
        M = cv2.getRotationMatrix2D((cx, cy), math.degrees(self.yaw), 1.0)
        # Rotate the entire map (inefficient but simple)
        rot_map = cv2.warpAffine(self.full_map, M, (self.map_w, self.map_h))
        
        x1 = cx - self.view_w_px // 2
        y1 = cy - self.view_h_px // 2
        
        # Extract crop with boundary checks
        crop = np.zeros((self.view_h_px, self.view_w_px, 3), dtype=np.uint8)
        
        # Safe slicing logic
        y1_src = max(0, y1); y2_src = min(self.map_h, y1 + self.view_h_px)
        x1_src = max(0, x1); x2_src = min(self.map_w, x1 + self.view_w_px)
        
        y1_dst = max(0, -y1); y2_dst = y1_dst + (y2_src - y1_src)
        x1_dst = max(0, -x1); x2_dst = x1_dst + (x2_src - x1_src)

        if y2_src > y1_src and x2_src > x1_src:
            crop[y1_dst:y2_dst, x1_dst:x2_dst] = rot_map[y1_src:y2_src, x1_src:x2_src]

        return cv2.resize(crop, (IMAGE_W, IMAGE_H))

    def get_god_view(self):
        """ Generates the Map View with Overlays (Left Side) """
        display_map = self.full_map.copy()
        cx, cy = int(self.x * self.pix_per_m), int(self.y * self.pix_per_m)
        
        # Draw Waypoints
        for i, wp in enumerate(self.waypoints):
            wx = int(wp[0] * self.pix_per_m)
            wy = int(wp[1] * self.pix_per_m)
            cv2.circle(display_map, (wx, wy), 3, (200, 200, 200), -1)
            if i > 0:
                prev_wp = self.waypoints[i-1]
                px = int(prev_wp[0] * self.pix_per_m)
                py = int(prev_wp[1] * self.pix_per_m)
                cv2.line(display_map, (px, py), (wx, wy), (100, 100, 100), 1)

        # Draw Drone
        cv2.circle(display_map, (cx, cy), 8, (255, 0, 0), -1)
        
        # Draw Camera Box
        rect = ((cx, cy), (self.view_w_px, self.view_h_px), math.degrees(self.yaw))
        box = cv2.boxPoints(rect) 
        box = np.int32(box)
        cv2.drawContours(display_map, [box], 0, (0, 255, 255), 2)
        
        # Resize
        target_h = IMAGE_H
        scale = target_h / self.map_h
        target_w = int(self.map_w * scale)
        return cv2.resize(display_map, (target_w, target_h))

    def run(self):
        print("SIM STARTED.")
        print("  Mode: AUTO (Press 'M' for Manual)")
        print("  Manual Controls: W/S (Fwd/Back), A/D (Strafe), Q/E (Yaw)")
        
        while True:
            # 1. Update Physics / Autopilot
            if self.mode == "AUTO":
                self.update_autopilot()
            
            # 2. Generate Views
            drone_frame = self.get_drone_view()
            god_frame = self.get_god_view()
            
            # 3. Vision Check
            found, u, v = self.eyes.process_frame_manually(drone_frame)
            
            geo_msg = "Scanning..."
            color = (0, 255, 255) # Yellow
            
            if found:
                self.target_found = True # Stop Autopilot
                
                # Math
                px, py = pixel_to_gps(u, v, self.alt, 0, 0, self.yaw, self.x, self.y)
                err = math.sqrt((px - self.target_pos[0])**2 + (py - self.target_pos[1])**2)
                
                geo_msg = f"TARGET LOCKED! Err: {err:.2f}m"
                color = (0, 255, 0) # Green
                
                cv2.circle(drone_frame, (u, v), 10, color, 2)
                cv2.line(drone_frame, (u-15, v), (u+15, v), color, 2)
                cv2.line(drone_frame, (u, v-15), (u, v+15), color, 2)

            # 4. HUD
            cv2.putText(drone_frame, f"Mode: {self.mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(drone_frame, f"Alt: {self.alt:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(drone_frame, geo_msg, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            combined = np.hstack((god_frame, drone_frame))
            cv2.imshow("AENGM0074 Drone Simulator", combined)
            
            # 5. Controls
            key = cv2.waitKey(30)
            if key == 27: break
            
            # Toggle Mode
            if key == ord('m'):
                self.mode = "MANUAL" if self.mode == "AUTO" else "AUTO"
                self.target_found = False # Reset stop flag if manually taking over
                print(f"Switched to {self.mode}")

            # Manual Flight Inputs
            if self.mode == "MANUAL":
                move_speed = 1.0 * (self.alt / 10.0)
                if move_speed < 1.0: move_speed = 1.0
                
                if key == ord('w'): 
                    self.x += math.cos(self.yaw) * move_speed
                    self.y += math.sin(self.yaw) * move_speed
                if key == ord('s'): 
                    self.x -= math.cos(self.yaw) * move_speed
                    self.y -= math.sin(self.yaw) * move_speed
                if key == ord('d'): 
                    self.x += math.cos(self.yaw + 1.57) * move_speed
                    self.y += math.sin(self.yaw + 1.57) * move_speed
                if key == ord('a'): 
                    self.x -= math.cos(self.yaw + 1.57) * move_speed
                    self.y -= math.sin(self.yaw + 1.57) * move_speed
                if key == ord('q'): self.yaw -= 0.1
                if key == ord('e'): self.yaw += 0.1
                if key == 82: self.alt += 1
                if key == 84: self.alt -= 1
                self.alt = max(5.0, self.alt)

if __name__ == "__main__":
    DroneSim().run()