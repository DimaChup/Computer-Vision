# filename: connection_test.py
from pymavlink import mavutil
import time
import sys

# --- CONFIG ---
# Try these ports one by one if it fails:
# 'tcp:127.0.0.1:5760' (Standard SITL port)
# 'tcp:127.0.0.1:5762' (First extra output)
# 'tcp:127.0.0.1:5763' (Second extra output)
# 'udp:127.0.0.1:14550' (Standard GCS bridge)
CONNECTION_STR = 'tcp:127.0.0.1:5762'

def test_connection():
    print(f"--- MAVLINK CONNECTION TESTER ---")
    print(f"Attempting to connect to: {CONNECTION_STR}")
    print(f"Waiting for Heartbeat... (Press Ctrl+C to stop)")

    try:
        # 1. Create Connection
        master = mavutil.mavlink_connection(CONNECTION_STR)
        
        # 2. Wait for the first heartbeat (This confirms connection)
        # This function blocks until a message is received
        master.wait_heartbeat(timeout=10)
        
        print("\n>>> SUCCESS! Heartbeat Received! <<<")
        print(f"Target System ID: {master.target_system}")
        print(f"Target Component ID: {master.target_component}")
        
        # 3. Request Data Stream (Just in case SITL is being quiet)
        # Request all data streams at 2 Hz
        print("Requesting Data Streams...")
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 2, 1
        )

        # 4. Listen Loop
        print("\nListening for Telemetry (GLOBAL_POSITION_INT)...")
        count = 0
        while True:
            # Check for specific position message
            msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
            
            if msg:
                count += 1
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.relative_alt / 1000.0
                sys.stdout.write(f"\rPacket #{count} | Lat: {lat:.6f} | Lon: {lon:.6f} | Alt: {alt:.1f}m   ")
                sys.stdout.flush()
            else:
                # If we timeout waiting for position, check if we are getting ANYTHING
                any_msg = master.recv_match(blocking=False)
                if any_msg:
                    print(f"\nReceived {any_msg.get_type()} (but waiting for POSITION...)")
                else:
                    print(".", end="", flush=True)
            
            time.sleep(0.1)

    except Exception as e:
        print(f"\nERROR: Connection Failed: {e}")
        print("Tip: Check if Mission Planner is running and if the port matches.")

if __name__ == "__main__":
    test_connection()