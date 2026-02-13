"""
Test motors one at a time at low throttle.
PROPS MUST BE OFF!
Run on Pi: python3 test_motors.py
"""
from pymavlink import mavutil
import time

PORT = '/dev/ttyAMA0'
BAUD = 921600

print("Connecting to Cube...")
conn = mavutil.mavlink_connection(PORT, baud=BAUD)
conn.wait_heartbeat(timeout=10)
print("Connected!")

# Request streams so we can see armed status
conn.mav.request_data_stream_send(
    conn.target_system, conn.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1
)

input("\nPROPS OFF? Press Enter to start motor test...")

# Test each motor (1-4) at 10% throttle for 2 seconds each
num_motors = 4
throttle_percent = 10
duration_sec = 2

for motor in range(1, num_motors + 1):
    print(f"\nSpinning Motor {motor} at {throttle_percent}% for {duration_sec}s...")

    # MAV_CMD_DO_MOTOR_TEST
    # param1 = motor instance (1-based)
    # param2 = throttle type (0 = percent)
    # param3 = throttle value (0-100)
    # param4 = timeout in seconds
    # param5 = motor count (0 = test one motor)
    # param6 = test order (0 = default)
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST, 0,
        motor,       # motor instance
        0,           # throttle type = percent
        throttle_percent,
        duration_sec,
        0, 0, 0
    )

    # Wait for it to finish + small gap
    time.sleep(duration_sec + 1)
    print(f"Motor {motor} done.")

print("\nAll motors tested!")
print("If none spun, the Cube may need battery voltage (not just USB).")
print("Check the power supply is providing 14.8V or 22.2V depending on your setup.")
conn.close()
