# filename: flight_control.py
from pymavlink import mavutil
import time

class FlightController:
    def __init__(self, connection_string='/dev/ttyAMA0', baud=57600):
        print(f"Connecting to Drone on {connection_string}...")
        try:
            self.master = mavutil.mavlink_connection(connection_string, baud=baud)
            self.master.wait_heartbeat(timeout=5)
            print("Drone Connected!")
        except Exception as e:
            print(f"ERROR: Could not connect to drone: {e}")
            self.master = None

    def get_telemetry(self):
        """ Returns lat, lon, alt, roll, pitch, yaw """
        if not self.master: return None
        pos = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.1)
        att = self.master.recv_match(type='ATTITUDE', blocking=True, timeout=0.1)
        
        if pos and att:
            return (pos.lat/1e7, pos.lon/1e7, pos.relative_alt/1000.0, 
                    att.roll, att.pitch, att.yaw)
        return None

    def set_mode(self, mode):
        if not self.master: return
        mode_id = self.master.mode_mapping()[mode]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)

    def takeoff(self, target_alt):
        if not self.master: return
        print(f"Taking off to {target_alt}m")
        self.set_mode('GUIDED')
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, target_alt)

    def goto_location(self, lat, lon, alt):
        if not self.master: return
        # Send Guided Waypoint
        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000, 
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0)

    def land_at_location_blind(self, lat, lon):
        if not self.master: return
        # Descend to 0m (fights wind)
        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000,
            int(lat * 1e7), int(lon * 1e7), 0,
            0, 0, 0, 0, 0, 0, 0, 0)
            
    def drop_payload(self):
        if not self.master: return
        print("DROPPING PAYLOAD")
        # Example: Servo Channel 9
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
            9, 1900, 0, 0, 0, 0, 0)