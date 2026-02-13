"""
Test 2: Cube Connection
Run this on the Pi to check if the Cube flight controller is connected.
Plug USB cable from Pi to Cube before running.
This is READ ONLY - nothing will move or arm.
"""
from pymavlink import mavutil
import time

# Try common device paths (UART first, then USB)
ports = ['/dev/ttyAMA0', '/dev/serial0', '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']

connection = None
for port in ports:
    try:
        print(f"Trying {port}...")
        connection = mavutil.mavlink_connection(port, baud=57600)
        print(f"Connected on {port}! Waiting for heartbeat...")
        connection.wait_heartbeat(timeout=10)
        print(f"Heartbeat received!")
        break
    except Exception as e:
        print(f"  Not found on {port}: {e}")
        connection = None

if connection is None:
    print("\nFAILED - Cube not found on any port.")
    print("Check:")
    print("  1. Is USB cable plugged in (Pi USB -> Cube USB)?")
    print("  2. Is the Cube powered on?")
    print("  3. Run: ls /dev/ttyACM* /dev/ttyUSB*  to see available ports")
    exit(1)

# Read some data
print(f"\nSystem ID: {connection.target_system}")
print(f"Component ID: {connection.target_component}")

# Request data and read a few messages
print("\nReading messages for 5 seconds...\n")
start = time.time()
msg_types_seen = set()

while time.time() - start < 5:
    msg = connection.recv_match(blocking=True, timeout=1)
    if msg:
        msg_type = msg.get_type()
        if msg_type not in msg_types_seen:
            msg_types_seen.add(msg_type)
            if msg_type == 'GPS_RAW_INT':
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                print(f"  GPS:      lat={lat:.6f}, lon={lon:.6f}, sats={msg.satellites_visible}")
            elif msg_type == 'SYS_STATUS':
                voltage = msg.voltage_battery / 1000.0
                print(f"  Battery:  {voltage:.1f}V")
            elif msg_type == 'HEARTBEAT':
                mode = mavutil.mode_string_v10(msg)
                print(f"  Mode:     {mode}")
                print(f"  Armed:    {msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED != 0}")
            elif msg_type == 'ATTITUDE':
                print(f"  Attitude: roll={msg.roll:.2f}, pitch={msg.pitch:.2f}, yaw={msg.yaw:.2f}")

print(f"\nMessage types received: {', '.join(sorted(msg_types_seen))}")
print("\nCube connection test PASSED!")
connection.close()
