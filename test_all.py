"""
Run all tests in one go on the Pi.
Usage: python3 test_all.py
"""
import time

print("=" * 50)
print("  TEST 1: Camera")
print("=" * 50)
try:
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration())
    cam.start()
    time.sleep(2)
    cam.capture_file("test_photo.jpg")
    cam.stop()
    print("CAMERA TEST PASSED - photo saved as test_photo.jpg")
except Exception as e:
    print(f"CAMERA TEST FAILED: {e}")

print()
print("=" * 50)
print("  TEST 2: Cube Connection (UART)")
print("=" * 50)
try:
    from pymavlink import mavutil

    ports = ['/dev/ttyAMA0', '/dev/serial0', '/dev/ttyACM0', '/dev/ttyUSB0']
    connection = None

    for port in ports:
        try:
            print(f"Trying {port}...")
            connection = mavutil.mavlink_connection(port, baud=57600)
            connection.wait_heartbeat(timeout=10)
            print(f"CONNECTED on {port}!")
            break
        except Exception:
            print(f"  No response on {port}")
            connection = None

    if connection:
        # Read messages for 5 seconds
        print("\nReading drone data...\n")
        start = time.time()
        seen = set()
        while time.time() - start < 5:
            msg = connection.recv_match(blocking=True, timeout=1)
            if msg:
                t = msg.get_type()
                if t not in seen:
                    seen.add(t)
                    if t == 'GPS_RAW_INT':
                        print(f"  GPS:      lat={msg.lat/1e7:.6f}, lon={msg.lon/1e7:.6f}, sats={msg.satellites_visible}")
                    elif t == 'SYS_STATUS':
                        print(f"  Battery:  {msg.voltage_battery/1000:.1f}V")
                    elif t == 'HEARTBEAT':
                        print(f"  Mode:     {mavutil.mode_string_v10(msg)}")
                        print(f"  Armed:    {msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED != 0}")
                    elif t == 'ATTITUDE':
                        print(f"  Attitude: roll={msg.roll:.2f}, pitch={msg.pitch:.2f}, yaw={msg.yaw:.2f}")

        print(f"\nMessage types: {', '.join(sorted(seen))}")
        print("CUBE TEST PASSED!")
        connection.close()
    else:
        print("\nCUBE TEST FAILED - not found on any port")
        print("Check: is the Cube powered? Are wires on TELEM2 pins 2,3,6 -> Pi pins 10,8,6?")

except Exception as e:
    print(f"CUBE TEST FAILED: {e}")

print()
print("=" * 50)
print("  TEST 3: List all serial ports")
print("=" * 50)
import glob
for pattern in ['/dev/ttyAMA*', '/dev/ttyS*', '/dev/ttyACM*', '/dev/ttyUSB*', '/dev/serial*']:
    matches = glob.glob(pattern)
    for m in matches:
        print(f"  Found: {m}")
if not any(glob.glob(p) for p in ['/dev/ttyAMA*', '/dev/ttyACM*', '/dev/ttyUSB*']):
    print("  No serial ports found!")

print()
print("ALL TESTS DONE")
