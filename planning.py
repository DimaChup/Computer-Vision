# planning.py
import math
import cv2
import numpy as np
import heapq
import config

class PathPlanner:
    def __init__(self, geo_transformer, search_poly, nfz_poly):
        self.geo = geo_transformer
        self.search_polygon = search_poly
        self.nfz_polygon = nfz_poly
        self.pix_per_m = self.geo.pix_per_m
        self.virtual_polygon = [] 

    # --- SAFETY & UTILS ---
    def check_geofence_violation(self, lat, lon):
        if len(self.nfz_polygon) < 3: return False, 999.0
        dx, dy = self.geo.gps_to_pixels(lat, lon)
        pts = np.array([self.nfz_polygon], dtype=np.int32)
        dist_px = cv2.pointPolygonTest(pts, (dx, dy), True)
        dist_m = dist_px / self.pix_per_m
        if dist_m > -config.NFZ_BUFFER_M:
            return True, dist_m
        return False, dist_m

    def plan_path_around_nfz(self, start_gps, end_gps):
        if len(self.nfz_polygon) < 3: return [end_gps]
        start_px = self.geo.gps_to_pixels(start_gps[0], start_gps[1])
        end_px = self.geo.gps_to_pixels(end_gps[0], end_gps[1])
        
        def intersect(A, B, C, D):
            def ccw(p1, p2, p3): return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        def line_blocked(p1, p2, poly):
            for i in range(len(poly)):
                p3 = poly[i]
                p4 = poly[(i+1)%len(poly)]
                if intersect(p1, p2, p3, p4): return True
            return False

        if not line_blocked(start_px, end_px, self.nfz_polygon): return [end_gps] 
            
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

        # Square off corners logic
        corner_start_idx = 2
        num_corners = len(nodes) - 2
        tolerance_px = 45.0 * self.pix_per_m 
        
        if num_corners > 1:
            def get_edge_dist(u, target_node):
                for d, v in adj[u]:
                    if v == target_node: return d
                return None
            for i in range(num_corners):
                u = corner_start_idx + i
                dist_u_end = get_edge_dist(u, 1)
                if dist_u_end is not None:
                    poly_idx = i
                    next_poly = (poly_idx + 1) % num_corners
                    prev_poly = (poly_idx - 1 + num_corners) % num_corners
                    neighbors = [corner_start_idx + next_poly, corner_start_idx + prev_poly]
                    for v in neighbors:
                        dist_v_end = get_edge_dist(v, 1)
                        dist_u_v = get_edge_dist(u, v)
                        if dist_v_end is not None and dist_u_v is not None:
                            if dist_v_end < (dist_u_end + tolerance_px):
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
                    lat, lon = self.geo.pixels_to_gps(px[0], px[1])
                    gps_path.append((lat, lon))
                return gps_path
            for weight, v in adj[u]:
                if v not in visited:
                    heapq.heappush(pq, (cost + weight, v, path))
        return [end_gps]

    # --- MAIN MANAGER ---
    def generate_search_pattern(self, map_w, map_h):
        """ 1. Prepares Safe Area. 2. Calls Selected Algorithm. """
        if len(self.search_polygon) < 3: return []

        print(f"[PLANNER] Generating Grid using: {config.SEARCH_ALGORITHM}")

        # 1. GENERATE SAFE MASK (Common to all algorithms)
        safe_mask = self._generate_safe_mask(map_w, map_h)
        if safe_mask is None: return []

        # 2. DISPATCH TO ALGORITHM
        if config.SEARCH_ALGORITHM == "ROTATED_SWEEP":
            return self._algo_rotated_sweep(safe_mask, map_w, map_h)
        
        elif config.SEARCH_ALGORITHM == "SIMPLE_LAWNMOWER":
            return self._algo_simple_lawnmower(safe_mask, map_w, map_h)
        
        else:
            print(f"Unknown Algorithm '{config.SEARCH_ALGORITHM}'. Defaulting to Rotated.")
            return self._algo_rotated_sweep(safe_mask, map_w, map_h)

    # --- PRE-PROCESSING ---
    def _generate_safe_mask(self, map_w, map_h):
        mask = np.zeros((map_h, map_w), dtype=np.uint8)
        poly_pts = np.array([self.search_polygon], dtype=np.int32)
        cv2.fillPoly(mask, poly_pts, 255)
        
        # Calculate Inset
        ground_width_m = (config.SENSOR_WIDTH_MM * config.TARGET_ALT) / config.FOCAL_LENGTH_MM
        inset_dist_m = 13.0 
        inset_px = int(inset_dist_m * self.pix_per_m)
        
        if inset_px > 0:
            kernel_size = 2 * inset_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask = cv2.erode(mask, kernel)
            
            # Update Virtual Polygon for Visualization
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                self.virtual_polygon = max(contours, key=cv2.contourArea)
            else:
                self.virtual_polygon = []
                print("Warning: Search area vanished after inset!")
                return None
        
        # Subtract NFZ
        if len(self.nfz_polygon) >= 3:
            nfz_mask = np.zeros_like(mask)
            nfz_pts = np.array([self.nfz_polygon], dtype=np.int32)
            cv2.fillPoly(nfz_mask, nfz_pts, 255)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(nfz_mask))
            
            # Re-update Virtual Polygon to match safe area
            safe_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if safe_contours:
                 self.virtual_polygon = max(safe_contours, key=cv2.contourArea)

        return mask

    # --- ALGORITHM 1: ROTATED SWEEP (Optimized) ---
    def _algo_rotated_sweep(self, mask, map_w, map_h):
        ground_width_m = (config.SENSOR_WIDTH_MM * config.TARGET_ALT) / config.FOCAL_LENGTH_MM
        
        # 1. Determine Angle
        poly_pts = np.array([self.search_polygon], dtype=np.int32)
        rect = cv2.minAreaRect(poly_pts[0])
        (center, size, angle) = rect
        scan_angle = angle + 90 if size[0] < size[1] else angle
        
        # 2. Rotate Mask
        M = cv2.getRotationMatrix2D(center, scan_angle, 1.0)
        M_inv = cv2.invertAffineTransform(M)
        rotated_mask = cv2.warpAffine(mask, M, (map_w, map_h))

        # 3. Scan Lines
        overlap = 0.2
        SWATH_M = ground_width_m * (1.0 - overlap)
        step_px = int(SWATH_M * self.pix_per_m)
        if step_px < 1: step_px = 1
        
        points = cv2.findNonZero(rotated_mask)
        if points is None: return []
        x, y, w, h = cv2.boundingRect(points)
        
        wps = []
        direction = 1 
        
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
                    segments.append((pixels[segment_starts[i]], pixels[segment_ends[i]]))
                
                if direction == -1: segments.reverse()
                for x_start, x_end in segments:
                    pt1 = np.array([[[x_start, scan_y]]], dtype=np.float32)
                    pt2 = np.array([[[x_end, scan_y]]], dtype=np.float32)
                    pt1_orig = cv2.transform(pt1, M_inv)[0][0]
                    pt2_orig = cv2.transform(pt2, M_inv)[0][0]
                    p1_gps = self.geo.pixels_to_gps(pt1_orig[0], pt1_orig[1])
                    p2_gps = self.geo.pixels_to_gps(pt2_orig[0], pt2_orig[1])
                    if direction == 1:
                        wps.extend([((p1_gps[0], p1_gps[1])), ((p2_gps[0], p2_gps[1]))])
                    else:
                        wps.extend([((p2_gps[0], p2_gps[1])), ((p1_gps[0], p1_gps[1]))])
                direction *= -1

        return self._add_perimeter(wps)

    # --- ALGORITHM 2: SIMPLE LAWNMOWER (No Rotation) ---
    def _algo_simple_lawnmower(self, mask, map_w, map_h):
        # Uses the mask directly without rotating it. Best for weird shapes.
        ground_width_m = (config.SENSOR_WIDTH_MM * config.TARGET_ALT) / config.FOCAL_LENGTH_MM
        
        overlap = 0.2
        SWATH_M = ground_width_m * (1.0 - overlap)
        step_px = int(SWATH_M * self.pix_per_m)
        if step_px < 1: step_px = 1
        
        points = cv2.findNonZero(mask)
        if points is None: return []
        x, y, w, h = cv2.boundingRect(points)
        
        wps = []
        direction = 1
        
        for scan_y in range(y + step_px//2, y + h, step_px):
            row = mask[scan_y, :]
            pixels = np.where(row == 255)[0]
            if len(pixels) > 0:
                # Find segments (in case NFZ cuts a line in half)
                diffs = np.diff(pixels)
                breaks = np.where(diffs > 1)[0]
                segment_starts = [0] + list(breaks + 1)
                segment_ends = list(breaks) + [len(pixels)-1]
                
                segments = []
                for i in range(len(segment_starts)):
                    segments.append((pixels[segment_starts[i]], pixels[segment_ends[i]]))
                
                if direction == -1: segments.reverse()
                
                for x_start, x_end in segments:
                    p1_gps = self.geo.pixels_to_gps(x_start, scan_y)
                    p2_gps = self.geo.pixels_to_gps(x_end, scan_y)
                    if direction == 1:
                        wps.extend([((p1_gps[0], p1_gps[1])), ((p2_gps[0], p2_gps[1]))])
                    else:
                        wps.extend([((p2_gps[0], p2_gps[1])), ((p1_gps[0], p1_gps[1]))])
                direction *= -1
                
        return self._add_perimeter(wps)

    def _add_perimeter(self, wps):
        # Helper to prepend the perimeter path
        perimeter_wps = []
        if len(self.virtual_polygon) > 0:
            epsilon = 0.005 * cv2.arcLength(self.virtual_polygon, True)
            approx = cv2.approxPolyDP(self.virtual_polygon, epsilon, True)
            cnt_pts = approx.reshape(-1, 2)
            for pt in cnt_pts:
                lat, lon = self.geo.pixels_to_gps(pt[0], pt[1])
                perimeter_wps.append((lat, lon))
            if len(perimeter_wps) > 0:
                perimeter_wps.append(perimeter_wps[0])
        return perimeter_wps + wps