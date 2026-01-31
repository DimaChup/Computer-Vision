# filename: main.py
import threading
import time
import math
from flight_control import FlightController
from vision import VisionSystem
from geolocation import pixel_to_gps

# --- MISSION CONFIG ---
SEARCH_ALTITUDE = 35.0  # Meters
LANDING_OFFSET = 7.5    # Meters North (approx 0.0000675 deg)
HOVER_SAMPLES = 20      # Detections before confirming

# --- SHARED STATE ---
class MissionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "SETUP" 
        self.target_found = False
        self.detections = [] 
        self.final_target = None
        # Telemetry
        self.lat = 0; self.lon = 0; self.alt = 0
        self.roll = 0; self.pitch = 0; self.yaw = 0

state = MissionState()

def vision_thread():
    print("[VISION] Started")
    eyes = VisionSystem(camera_index=0) # Use Real Camera
    while True:
        if state.phase in ["SEARCH", "VERIFY"]:
            found, u, v = eyes.get_latest_detection()
            if found:
                with state.lock:
                    # Snapshot telemetry
                    c_lat, c_lon, c_alt = state.lat, state.lon, state.alt
                    c_roll, c_pitch, c_yaw = state.roll, state.pitch, state.yaw
                
                t_lat, t_lon = pixel_to_gps(u, v, c_alt, c_roll, c_pitch, c_yaw, c_lat, c_lon)
                
                if t_lat != 0.0:
                    with state.lock:
                        state.target_found = True
                        if state.phase == "VERIFY":
                            state.detections.append((t_lat, t_lon))
                            print(f"[VISION] Sample: {t_lat:.6f}, {t_lon:.6f}")
        time.sleep(0.05)

def flight_thread():
    print("[FLIGHT] Started")
    drone = FlightController() # Connect to Cube
    
    while True:
        # 1. Update Telemetry
        telemetry = drone.get_telemetry()
        if telemetry:
            with state.lock:
                state.lat, state.lon, state.alt, state.roll, state.pitch, state.yaw = telemetry
        
        # 2. State Machine
        if state.phase == "SETUP":
            # Wait for user to trigger mission (e.g. switch to GUIDED)
            pass

        elif state.phase == "TAKEOFF":
            drone.takeoff(SEARCH_ALTITUDE)
            if state.alt >= SEARCH_ALTITUDE - 1.0:
                print("Altitude Reached. SEARCHING.")
                state.phase = "SEARCH"

        elif state.phase == "SEARCH":
            # (Add Waypoint/Lawnmower logic here)
            if state.target_found:
                print("TARGET SPOTTED. LOITERING.")
                drone.set_mode("LOITER")
                time.sleep(2)
                state.phase = "VERIFY"

        elif state.phase == "VERIFY":
            if len(state.detections) >= HOVER_SAMPLES:
                # Average GPS
                avg_lat = sum(d[0] for d in state.detections) / len(state.detections)
                avg_lon = sum(d[1] for d in state.detections) / len(state.detections)
                state.final_target = (avg_lat, avg_lon)
                print(f"TARGET CONFIRMED: {avg_lat}, {avg_lon}")
                state.phase = "CALCULATE"

        elif state.phase == "CALCULATE":
            t_lat, t_lon = state.final_target
            # Add Offset North
            land_lat = t_lat + 0.0000675
            land_lon = t_lon
            state.final_target = (land_lat, land_lon)
            print(f"LANDING AT OFFSET: {land_lat}, {land_lon}")
            state.phase = "APPROACH"

        elif state.phase == "APPROACH":
            t = state.final_target
            drone.goto_location(t[0], t[1], SEARCH_ALTITUDE)
            # Distance check
            dist = math.sqrt((state.lat - t[0])**2 + (state.lon - t[1])**2) * 111111
            if dist < 1.0:
                print("ALIGNED. DESCENDING BLIND.")
                state.phase = "LANDING"

        elif state.phase == "LANDING":
            t = state.final_target
            drone.land_at_location_blind(t[0], t[1])
            if state.alt < 0.5:
                print("TOUCHDOWN. DROPPING PAYLOAD.")
                drone.drop_payload()
                state.phase = "DONE"
                
        time.sleep(0.1)

if __name__ == "__main__":
    t1 = threading.Thread(target=vision_thread); t1.start()
    t2 = threading.Thread(target=flight_thread); t2.start()