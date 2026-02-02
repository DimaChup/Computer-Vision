# planning.py
import math
import cv2
import numpy as np
import config

class PathPlanner:
    def __init__(self, geo_transformer, search_poly):
        self.geo = geo_transformer
        self.search_polygon = search_poly
        self.pix_per_m = self.geo.pix_per_m
        self.virtual_polygon = [] 

    # --- MAIN GENERATOR (OPTIMIZED) ---
    def generate_search_pattern(self, map_w, map_h, drone_gps=None):
        if len(self.search_polygon) < 3: return []

        print("[PLANNER] Calculating Optimum Path (Full Green Polygon Coverage)")

        # 1. GENERATE SEARCH MASK
        mask = np.zeros((map_h, map_w), dtype=np.uint8)
        poly_pts = np.array([self.search_polygon], dtype=np.int32)
        cv2.fillPoly(mask, poly_pts, 255)
        
        # Update Virtual Polygon for Visualization (It is now exactly the Search Poly)
        self.virtual_polygon = poly_pts.reshape(-1, 1, 2)

        # 2. ROTATED SWEEP LOGIC
        
        # Determine Angle (Principal Component Analysis or MinAreaRect)
        rect = cv2.minAreaRect(poly_pts[0])
        (center, size, angle) = rect
        scan_angle = angle + 90 if size[0] < size[1] else angle
        
        # Rotate Mask to align with longest edge
        M = cv2.getRotationMatrix2D(center, scan_angle, 1.0)
        M_inv = cv2.invertAffineTransform(M)
        rotated_mask = cv2.warpAffine(mask, M, (map_w, map_h))

        # Scan Lines
        ground_width_m = (config.SENSOR_WIDTH_MM * config.TARGET_ALT) / config.FOCAL_LENGTH_MM
        overlap = 0.2
        SWATH_M = ground_width_m * (1.0 - overlap)
        step_px = int(SWATH_M * self.pix_per_m)
        if step_px < 1: step_px = 1
        
        points = cv2.findNonZero(rotated_mask)
        if points is None: return []
        x, y, w, h = cv2.boundingRect(points)
        
        # Generate ALL possible lines (Strips)
        all_strips = []
        
        for scan_y in range(y + step_px//2, y + h, step_px):
            row = rotated_mask[scan_y, :]
            pixels = np.where(row == 255)[0]
            if len(pixels) > 0:
                # Find start and end of the scan line in this row
                x_start = pixels[0]
                x_end = pixels[-1]
                # Store as rotated points (x, y)
                all_strips.append( [ (x_start, scan_y), (x_end, scan_y) ] )

        # 3. GLOBAL OPTIMIZATION: CLOSEST START CORNER
        direction = 1 # Default (Left to Right)
        
        if drone_gps and all_strips:
            drone_px = self.geo.gps_to_pixels(drone_gps[0], drone_gps[1])
            
            def get_dist_to_rotated_point(p_rot):
                p_arr = np.array([[p_rot]], dtype=np.float32)
                p_orig = cv2.transform(p_arr, M_inv)[0][0]
                return (p_orig[0]-drone_px[0])**2 + (p_orig[1]-drone_px[1])**2

            first_strip = all_strips[0]
            last_strip = all_strips[-1]
            
            # Check 4 Corners
            d_top_left = get_dist_to_rotated_point(first_strip[0])
            d_top_right = get_dist_to_rotated_point(first_strip[1])
            d_bot_left = get_dist_to_rotated_point(last_strip[0])
            d_bot_right = get_dist_to_rotated_point(last_strip[1])
            
            min_dist = min(d_top_left, d_top_right, d_bot_left, d_bot_right)
            
            # Decide Top vs Bottom
            if min_dist == d_bot_left or min_dist == d_bot_right:
                all_strips.reverse()
                # If we swapped, the "first strip" is now the bottom one
                # Re-eval left/right for the NEW first strip
                d_left = get_dist_to_rotated_point(all_strips[0][0])
                d_right = get_dist_to_rotated_point(all_strips[0][1])
            else:
                d_left = d_top_left
                d_right = d_top_right
                
            # Decide Left vs Right
            if d_right < d_left:
                direction = -1

        # 4. CONVERT TO GPS WAYPOINTS
        wps = []
        for strip in all_strips:
            # strip is [(x1, y), (x2, y)]
            if direction == -1:
                pt_start, pt_end = strip[1], strip[0]
            else:
                pt_start, pt_end = strip[0], strip[1]
            
            pts_rot = np.array([[pt_start, pt_end]], dtype=np.float32)
            pts_orig = cv2.transform(pts_rot, M_inv)[0]
            
            p1_gps = self.geo.pixels_to_gps(pts_orig[0][0], pts_orig[0][1])
            p2_gps = self.geo.pixels_to_gps(pts_orig[1][0], pts_orig[1][1])
            
            wps.append(p1_gps)
            wps.append(p2_gps)
            
            # ZigZag
            direction *= -1

        return wps