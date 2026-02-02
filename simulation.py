# simulation.py
import cv2
import numpy as np
import math
import config
from utils import overlay_image_alpha

class SimulationEnvironment:
    def __init__(self, geo_transformer):
        self.geo = geo_transformer
        
        # Load Map
        self.full_map = cv2.imread(config.MAP_FILE)
        if self.full_map is None:
            print(f"Error: '{config.MAP_FILE}' not found. Creating a fake green map.")
            self.full_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
            self.full_map[:] = (34, 139, 34)
            cv2.line(self.full_map, (0,0), (1000,1000), (255,255,255), 2)
        
        self.map_h, self.map_w = self.full_map.shape[:2]
        self.coverage_overlay = np.zeros_like(self.full_map)
        
        # Load Dummy Asset
        self.dummy_img = cv2.imread(config.DUMMY_FILE, cv2.IMREAD_UNCHANGED)
        if self.dummy_img is None:
            print(f"Warning: '{config.DUMMY_FILE}' not found. Stickman mode only.")
            
        # Calc Sizes
        raw_radius = config.TARGET_REAL_RADIUS_M * self.geo.pix_per_m
        self.target_radius_px = max(2, int(raw_radius))
        print(f"Map Scale: 1m = {self.geo.pix_per_m:.2f} px")
        
        # Sim State
        self.sim_target_type = None
        self.sim_target_px = None
        
    def setup_on_map(self):
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

        temp_type = "dot"
        search_polygon = [] 
        polygon_closed = False
        target_px = None
        
        window_name = "Select Target" 

        def mouse_callback(event, x, y, flags, param):
            nonlocal target_px, temp_type, polygon_closed
            real_x = int(x / scale_factor)
            real_y = int(y / scale_factor)
            update = False

            if event == cv2.EVENT_LBUTTONDOWN:
                if target_px is None:
                    target_px = (real_x, real_y)
                    if flags & cv2.EVENT_FLAG_CTRLKEY: temp_type = "dummy"
                    else: temp_type = "dot"
                    print("Step 1: Target Placed. Now Left-Click to draw SEARCH Polygon.")
                elif not polygon_closed:
                    search_polygon.append((real_x, real_y))
                update = True

            elif event == cv2.EVENT_RBUTTONDOWN:
                if not polygon_closed and len(search_polygon) >= 3:
                    polygon_closed = True
                    print("Step 2: Polygon Closed. Press KEY to Launch.")
                    update = True

            if update:
                temp_vis = display_map.copy()
                if target_px:
                      sx, sy = int(target_px[0]*scale_factor), int(target_px[1]*scale_factor)
                      if temp_type == "dummy":
                          map_h_px = config.DUMMY_HEIGHT_M * self.geo.pix_per_m
                          display_h_px = max(20, int(map_h_px * scale_factor))
                          overlay_image_alpha(temp_vis, self.dummy_img, sx, sy, 0, display_h_px)
                      else:
                          vis_radius = max(2, int(self.target_radius_px * scale_factor))
                          cv2.circle(temp_vis, (sx, sy), vis_radius, (0, 0, 255), -1)

                if len(search_polygon) > 0:
                    pts = [np.array([[int(p[0]*scale_factor), int(p[1]*scale_factor)] for p in search_polygon], dtype=np.int32)]
                    cv2.polylines(temp_vis, pts, polygon_closed, (0, 255, 0), 2)
                    for p in pts[0]: cv2.circle(temp_vis, tuple(p), 3, (0, 255, 0), -1)
                
                cv2.imshow(window_name, temp_vis)

        cv2.namedWindow(window_name)
        cv2.imshow(window_name, display_map)
        cv2.setMouseCallback(window_name, mouse_callback)
        print("--- MISSION SETUP ---")
        print("1. Left-Click Target.")
        print("2. Left-Click Green Search Points -> Right-Click to Close.")
        print("3. Press KEY to start.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        
        self.sim_target_px = target_px
        self.sim_target_type = temp_type
        return target_px, temp_type, search_polygon

    def get_drone_view(self, cx, cy, alt, yaw):
        fov = 2 * math.atan(config.SENSOR_WIDTH_MM / (2 * config.FOCAL_LENGTH_MM))
        safe_alt = max(1.0, alt)
        ground_w = 2 * safe_alt * math.tan(fov / 2)
        view_w_px = int(ground_w * self.geo.pix_per_m)
        view_h_px = int(view_w_px * (config.IMAGE_H / config.IMAGE_W))
        
        diag = int(math.sqrt(view_w_px**2 + view_h_px**2))
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
        M = cv2.getRotationMatrix2D(center, math.degrees(yaw), 1.0)
        rotated_patch = cv2.warpAffine(raw_crop, M, (diag, diag))
        
        start_x = (diag - view_w_px) // 2
        start_y = (diag - view_h_px) // 2
        crop = rotated_patch[start_y:start_y+view_h_px, start_x:start_x+view_w_px]
        
        final_view = cv2.resize(crop, (config.IMAGE_W, config.IMAGE_H))
            
        if self.sim_target_px is not None:
            dx = self.sim_target_px[0] - cx
            dy = self.sim_target_px[1] - cy
            angle_rad = -yaw
            dx_rot = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            dy_rot = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            scale = config.IMAGE_W / max(1, view_w_px)
            screen_x = int((config.IMAGE_W / 2) + (dx_rot * scale))
            screen_y = int((config.IMAGE_H / 2) + (dy_rot * scale))
            px_per_m_screen = config.IMAGE_W / ground_w
            
            if self.sim_target_type == "dummy" and self.dummy_img is not None:
                dummy_h_screen = int(config.DUMMY_HEIGHT_M * px_per_m_screen)
                overlay_image_alpha(final_view, self.dummy_img, screen_x, screen_y, 0, dummy_h_screen, rotation_deg=math.degrees(yaw))
            else:
                dot_rad_screen = int(config.TARGET_REAL_RADIUS_M * px_per_m_screen)
                cv2.circle(final_view, (screen_x, screen_y), max(3, dot_rad_screen), (0, 0, 255), -1)

        return final_view, view_w_px, view_h_px

    def get_god_view(self, cx, cy, yaw, view_w_px, view_h_px, zoom_level, virtual_poly, search_poly, target_gps, landing_gps, geo_tool):
        display_map = self.full_map.copy()
        
        if self.sim_target_px is not None:
            if self.sim_target_type == "dummy" and self.dummy_img is not None:
                map_h_px = int(config.DUMMY_HEIGHT_M * self.geo.pix_per_m)
                map_h_px = max(10, map_h_px) 
                overlay_image_alpha(display_map, self.dummy_img, self.sim_target_px[0], self.sim_target_px[1], 0, map_h_px)
            else:
                cv2.circle(display_map, self.sim_target_px, self.target_radius_px, (0, 0, 255), -1)

        # Draw Polygons
        if len(search_poly) > 1:
              cv2.polylines(display_map, [np.array(search_poly, np.int32)], True, (0, 255, 0), 2)
        if len(virtual_poly) > 0:
              cv2.drawContours(display_map, [virtual_poly], -1, (255, 0, 255), 2)

        # Draw Coverage
        rect = ((cx, cy), (view_w_px, view_h_px), math.degrees(yaw))
        box = np.int32(cv2.boxPoints(rect))
        cv2.fillPoly(self.coverage_overlay, [box], (255, 255, 0)) 
        cv2.addWeighted(self.coverage_overlay, 0.2, display_map, 1.0, 0, display_map)
        
        cv2.circle(display_map, (cx, cy), 8, (255, 0, 0), -1)
        cv2.drawContours(display_map, [box], 0, (0, 255, 255), 2)
        
        if target_gps[0] != 0:
            tx, ty = geo_tool.gps_to_pixels(target_gps[0], target_gps[1])
            cv2.circle(display_map, (tx, ty), 5, (0, 255, 0), -1) 
        
        if landing_gps[0] != 0:
            lx, ly = geo_tool.gps_to_pixels(landing_gps[0], landing_gps[1])
            cv2.circle(display_map, (lx, ly), 5, (255, 0, 255), -1) 
            cv2.circle(display_map, (lx, ly), 20, (255, 255, 255), 1)

        # Apply Zoom
        if zoom_level > 1.0:
            h, w = display_map.shape[:2]
            crop_h = int(h / zoom_level)
            crop_w = int(w / zoom_level)
            x1 = max(0, min(w - crop_w, cx - crop_w // 2))
            y1 = max(0, min(h - crop_h, cy - crop_h // 2))
            x2 = x1 + crop_w; y2 = y1 + crop_h
            display_map = display_map[y1:y2, x1:x2]

        target_h = config.IMAGE_H
        base_scale = target_h / self.map_h
        target_w = int(self.map_w * base_scale)
        if zoom_level > 1.0: return cv2.resize(display_map, (config.IMAGE_H, config.IMAGE_H))
        return cv2.resize(display_map, (target_w, target_h))