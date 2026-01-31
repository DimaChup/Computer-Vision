from pymavlink import mavutil
import time
import math
import sys

# --- CONFIG ---
CONNECTION_STR = 'tcp:127.0.0.1:5762'
TARGET_ALT = 35.0 # Meters
MOVE_DIST = 20.0  # Meters (Distance to fly East)

# --- STATES ---
class State:
    INIT = "INIT"
    CONNECTING = "CONNECTING"
    ARMING = "ARMING"
    TAKEOFF = "TAKEOFF"
    TRANSIT = "TRANSIT"
    LANDING = "LANDING"
    DONE = "DONE"

class FlightMission:
    def __init__(self):
        self.state = State.INIT
        self.master = None
        self.boot_time = time.time()
        self.last_req = 0
        self.last_heartbeat = 0
        
        # Telemetry storage
        self.lat = 0
        self.lon = 0
        self.alt = 0
        self.target_lat = 0
        self.target_lon = 0

    def run(self):
        print("Starting Mission State Machine...")
        while True:
            # 1. READ SENSORS (Non-blocking)
            self.update_telemetry()

            # 2. STATE LOGIC
            if self.state == State.INIT:
                print(f"[{self.state}] Connecting to {CONNECTION_STR}...")
                self.master = mavutil.mavlink_connection(CONNECTION_STR)
                self.state = State.CONNECTING

            elif self.state == State.CONNECTING:
                # Check if update_telemetry saw a heartbeat
                if self.last_heartbeat > 0:
                    print(f"[{self.state}] Heartbeat Received! System ID: {self.master.target_system}")
                    # Request data stream
                    self.master.mav.request_data_stream_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1
                    )
                    self.state = State.ARMING
                else:
                    # Print dot every second to show life
                    if time.time() - self.last_req > 1.0:
                        print(".", end="", flush=True)
                        self.last_req = time.time()

            elif self.state == State.ARMING:
                # Switch to GUIDED and ARM
                if time.time() - self.last_req > 2.0:
                    print(f"\n[{self.state}] Sending Arm Command...")
                    
                    # Set Mode GUIDED
                    mode_id = self.master.mode_mapping()['GUIDED']
                    self.master.mav.set_mode_send(
                        self.master.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        mode_id)
                    
                    # Send Arm
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                        0, 1, 0, 0, 0, 0, 0, 0)
                    
                    self.last_req = time.time()

                # Check if armed
                if self.master.motors_armed():
                    print(f"\n[{self.state}] Motors ARMED! Taking off.")
                    self.state = State.TAKEOFF
                    # Send Takeoff immediately
                    self.master.mav.command_long_send(
                        self.master.target_system, self.master.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                        0, 0, 0, 0, 0, 0, 0, TARGET_ALT)

            elif self.state == State.TAKEOFF:
                # Monitor Altitude
                if self.alt >= TARGET_ALT * 0.90:
                    print(f"\n[{self.state}] Target Altitude Reached ({self.alt:.1f}m).")
                    self.calculate_target()
                    self.state = State.TRANSIT
                else:
                    # Retry takeoff command every 5s if stuck
                    if time.time() - self.last_req > 5.0:
                        print(f"[{self.state}] Climbing... {self.alt:.1f}m / {TARGET_ALT}m")
                        self.master.mav.command_long_send(
                            self.master.target_system, self.master.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                            0, 0, 0, 0, 0, 0, 0, TARGET_ALT)
                        self.last_req = time.time()

            elif self.state == State.TRANSIT:
                # Send position target repeatedly (every 2s) to ensure it fights wind
                if time.time() - self.last_req > 2.0:
                    self.send_goto(self.target_lat, self.target_lon, TARGET_ALT)
                    self.last_req = time.time()

                # Check Distance
                dist = self.get_distance_to_target()
                sys.stdout.write(f"\r[{self.state}] Distance to Target: {dist:.1f}m   ")
                sys.stdout.flush()

                if dist < 1.0:
                    print(f"\n[{self.state}] Arrived! Landing.")
                    self.state = State.LANDING
                    # Send Land Command
                    mode_id = self.master.mode_mapping()['LAND']
                    self.master.mav.set_mode_send(
                        self.master.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        mode_id)

            elif self.state == State.LANDING:
                if self.alt < 0.5 or not self.master.motors_armed():
                    print(f"\n[{self.state}] Touchdown detected.")
                    self.state = State.DONE

            elif self.state == State.DONE:
                print("\nMission Complete.")
                return

            time.sleep(0.1) # Run loop at 10Hz

    def update_telemetry(self):
        if not self.master: return
        
        # Drain all packets from buffer to get latest
        while True:
            msg = self.master.recv_match(blocking=False)
            if not msg: break
            
            msg_type = msg.get_type()
            
            if msg_type == 'GLOBAL_POSITION_INT':
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.alt = msg.relative_alt / 1000.0
            
            elif msg_type == 'HEARTBEAT':
                self.last_heartbeat = time.time()

    def calculate_target(self):
        # 20m East calculation
        print(f"Calculating Target {MOVE_DIST}m East...")
        meters_per_deg_lon = 111132.0 * math.cos(math.radians(self.lat))
        delta_lon = MOVE_DIST / meters_per_deg_lon
        self.target_lat = self.lat
        self.target_lon = self.lon + delta_lon
        print(f"Target: {self.target_lat}, {self.target_lon}")

    def send_goto(self, lat, lon, alt):
        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000, 
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0)

    def get_distance_to_target(self):
        meters_per_deg_lon = 111132.0 * math.cos(math.radians(self.lat))
        d_lat = (self.lat - self.target_lat) * 111132
        d_lon = (self.lon - self.target_lon) * meters_per_deg_lon
        return math.sqrt(d_lat**2 + d_lon**2)

if __name__ == "__main__":
    mission = FlightMission()
    mission.run()