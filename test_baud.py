"""
Test which baud rate works for the Cube on TELEM2.
Run on Pi: python3 test_baud.py
"""
from pymavlink import mavutil

ports = ['/dev/ttyAMA0', '/dev/serial0', '/dev/ttyACM0', '/dev/ttyUSB0']
bauds = [115200, 921600, 57600, 38400]
found = False

for port in ports:
    if found:
        break
    for baud in bauds:
        try:
            print(f"Trying {port} at {baud}...")
            conn = mavutil.mavlink_connection(port, baud=baud)
            conn.wait_heartbeat(timeout=5)
            msg = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
            if msg:
                print(f"\n  WORKS! Port: {port}, Baud: {baud}")
                print(f"  Mode: {mavutil.mode_string_v10(msg)}")
                conn.close()
                found = True
                break
            conn.close()
        except Exception:
            pass

if not found:
    print("No working combination found")

print("Done")
