# main.py
from pymavlink import mavutil
import time
import math
import cv2
import numpy as np
from vision import VisionSystem

# Import Modular Scripts
import config
from states import State
from utils import GeoTransformer
from planning import PathPlanner
from simulation import SimulationEnvironment

class VisualFlightMission:
    def __init__(self):
        # 1. Initialize Helpers
        self.geo = GeoTransformer(map_w_px=4800) # Temp init, sim will update
        self.sim = SimulationEnvironment(self.geo)
        
        # Update Geo based on actual map loaded in sim
        self.geo = GeoTransformer(map_w_px=self.sim.map_w)
        self.sim.geo = self.geo # Sync back

        # 2. Setup Simulation (User Clicks)
        self.sim_target_px, self.sim_target_type, self.search_polygon, self.nfz_polygon, self.manual_transit_pixels = self.sim.setup_on_map()
        
        # 3. Initialize Planner
        self.planner = PathPlanner(self.geo, self.search_polygon, self.nfz_polygon)

        # 4. Vision System
        self.eyes = VisionSystem(camera_index=None, model_path="best.tflite")
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
        self.lat = config.REF_LAT
        self.lon = config.REF_LON
        self.alt = 0.0
        self.vx = 0.0; self.vy = 0.0; self.vz = 0.0
        self.roll = 0; self.pitch = 0; self.yaw = 0
        self.current_conf = 0.0
        
        # 7. Mission Data
        self.waypoints = []
        self.wp_index = 0
        self.transit_waypoints = []
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
        self.last_speed_req = 0 

    def on_dashboard_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0: self.zoom_level = min(self.zoom_level * 1.2, 20.0)
            else: self.zoom_level = max(self.zoom_level / 1.2, 1.0)

    def update_dashboard(self):
        px, py = self.geo.gps_to_pixels(self.lat, self.lon)
        
        drone_frame, self.view_w_px, self.view_h_px = self.sim.get_drone_view(px, py, self.alt, self.yaw)
        
        god_frame = self.sim.get_god_view(
            px, py, self.yaw, self.view_w_px, self.view_h_px, self.zoom_level,
            self.planner.virtual_polygon, self.search_polygon, self.nfz_polygon,
            (self.target_lat, self.target_lon), (self.landing_lat, self.landing_lon), self.geo
        )
        
        cx, cy = config.IMAGE_W // 2, config.IMAGE_H // 2
        cv2.line(drone_frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 0), 2)
        cv2.line(drone_frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 0), 2)
        
        found, u, v, conf = self.eyes.process_frame_manually(drone_frame)
        self.current_conf = conf

        if found:
            cv2.circle(drone_frame, (u, v), 10, (0, 255, 0), 2)
            cv2.line(drone_frame, (u, v), (cx, cy), (0, 255, 0), 1) 
            cv2.putText(drone_frame, f"TARGET ({conf*100:.0f}%)", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Dashboard Text
        cv2.putText(drone_frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(drone_frame, f"Alt: {self.alt:.1f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        fov_w_m = (config.SENSOR_WIDTH_MM * max(1.0, self.alt)) / config.FOCAL_LENGTH_MM
        fov_h_m = fov_w_m * (config.IMAGE_H / config.IMAGE_W)
        cv2.putText(drone_frame, f"FOV: {fov_w_m:.1f}m x {fov_h_m:.1f}m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
                self.vx = msg.vx / 100.0 
                self.vy = msg.vy / 100.0 
                self.vz = msg.vz / 100.0 
            elif msg.get_type() == 'ATTITUDE':
                self.roll = msg.roll
                self.pitch = msg.pitch
                self.yaw = msg.yaw
            elif msg.get_type() == 'HEARTBEAT':
                self.last_heartbeat = time.time()

    def calculate_target_gps(self, u, v):
        Cx = config.IMAGE_W / 2; Cy = config.IMAGE_H / 2
        gsd_m = (config.SENSOR_WIDTH_MM * self.alt) / (config.FOCAL_LENGTH_MM * config.IMAGE_W)
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

    def set_speed(self, speed_mps):
        if time.time() - self.last_speed_req < 3.0: return
        print(f"[CMD] Setting Ground Speed to {speed_mps} m/s")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0, 1, speed_mps, -1, 0, 0, 0, 0
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
            
            # --- KEY INPUTS ---
            if key == ord('m') or key == ord('M'):
                if self.state != State.MANUAL:
                    print(f"[{self.state}] SWITCHING TO MANUAL CONTROL.")
                    if self.state != State.SAFETY_HALT:
                        self.previous_state = self.state 
                    self.state = State.MANUAL
                else:
                    print(f"[{self.state}] EXITING MANUAL.")
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
            violation, dist = self.planner.check_geofence_violation(self.lat, self.lon)
            if violation and self.state != State.MANUAL:
                if self.state != State.SAFETY_HALT:
                    print(f"[SAFETY] NFZ PROXIMITY ({dist:.1f}m). AUTOMATIC BRAKE.")
                    self.previous_state = self.state 
                    self.state = State.SAFETY_HALT
            
            # --- FLIGHT STATE MACHINE ---
            if self.state == State.SAFETY_HALT:
                 self.master.mav.set_position_target_local_ned_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                        0b110111000111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                 if key == ord('m') or key == ord('M'):
                      self.state = State.MANUAL

            elif self.state == State.INIT:
                if time.time() - self.last_req > 1.0:
                    try:
                        self.master = mavutil.mavlink_connection(config.CONNECTION_STR)
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
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, config.TARGET_ALT)

            elif self.state == State.TAKEOFF:
                if self.alt >= config.TARGET_ALT * 0.90:
                    self.waypoints = self.planner.generate_search_pattern(self.sim.map_w, self.sim.map_h)
                    
                    if len(self.manual_transit_pixels) > 0:
                        print(f"Loading {len(self.manual_transit_pixels)} Manual Transit Points...")
                        self.transit_waypoints = []
                        for px in self.manual_transit_pixels:
                            lat, lon = self.geo.pixels_to_gps(px[0], px[1])
                            self.transit_waypoints.append((lat, lon))
                        
                        if len(self.waypoints) > 0:
                            self.transit_waypoints.append(self.waypoints[0])
                            
                        self.transit_index = 0
                        self.state = State.TRANSIT_TO_SEARCH
                        self.last_speed_req = 0 
                    
                    elif self.waypoints:
                        self.transit_waypoints = self.planner.plan_path_around_nfz(
                            (self.lat, self.lon), self.waypoints[0])
                        self.transit_index = 0
                        self.state = State.TRANSIT_TO_SEARCH
                        self.last_speed_req = 0 
                    else:
                        self.state = State.SEARCH
                        self.last_speed_req = 0 

                elif time.time() - self.last_req > 5.0:
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, config.TARGET_ALT)
                    self.last_req = time.time()
            
            elif self.state == State.TRANSIT_TO_SEARCH:
                self.set_speed(config.TRANSIT_SPEED_MPS)

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
                            int(target[0] * 1e7), int(target[1] * 1e7), config.TARGET_ALT,
                            0, 0, 0, 0, 0, 0, 0, 0)
                        self.last_req = time.time()
                else:
                     print(f"[{self.state}] Arrived at Search Grid.")
                     self.state = State.SEARCH
                     self.last_speed_req = 0 

            elif self.state == State.SEARCH:
                self.set_speed(config.SEARCH_SPEED_MPS)

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
                            int(target[0] * 1e7), int(target[1] * 1e7), config.TARGET_ALT,
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
                        int(self.target_lat * 1e7), int(self.target_lon * 1e7), config.TARGET_ALT,
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
                        int(self.target_lat * 1e7), int(self.target_lon * 1e7), config.VERIFY_ALT, 
                        0, 0, 0, 0, 0, 0, 0, 0)
                    self.last_req = time.time()
                
                if self.alt <= (config.VERIFY_ALT + 1.0): 
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
                            int(self.target_lat * 1e7), int(self.target_lon * 1e7), config.VERIFY_ALT,
                            0, 0, 0, 0, 0, 0, 0, 0)

            elif self.state == State.APPROACH:
                if time.time() - self.last_req > 2.0:
                    self.master.mav.set_position_target_global_int_send(
                        0, self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                        0b110111111000, 
                        int(self.landing_lat * 1e7), int(self.landing_lon * 1e7), config.VERIFY_ALT, 
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