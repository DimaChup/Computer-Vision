"""
Step-by-step motor troubleshooting.
Run on Pi: python3 test_troubleshoot.py
"""
from pymavlink import mavutil
import time

PORT = '/dev/ttyAMA0'
BAUD = 921600

print("Connecting to Cube...")
conn = mavutil.mavlink_connection(PORT, baud=BAUD)
conn.wait_heartbeat(timeout=10)
print("Connected!\n")

# Request data streams
conn.mav.request_data_stream_send(
    conn.target_system, conn.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1
)
time.sleep(2)

# ===== STEP 1: Check battery voltage =====
print("=" * 50)
print("  STEP 1: POWER CHECK")
print("=" * 50)
msg = conn.recv_match(type='SYS_STATUS', blocking=True, timeout=5)
if msg:
    voltage = msg.voltage_battery / 1000
    current = msg.current_battery / 100
    print(f"  Voltage: {voltage:.1f}V")
    print(f"  Current: {current:.1f}A")
    if voltage < 5:
        print("  WARNING: Voltage too low! ESCs need 14.8V (4S) or 22.2V (6S)")
        print("  -> Set your power supply higher")
    elif voltage < 12:
        print("  WARNING: Voltage seems low. Check if this is a 4S or 6S drone")
    else:
        print("  OK - voltage looks good")
else:
    print("  No battery data received - power module may not be connected")

input("\nPress Enter for Step 2...")

# ===== STEP 2: Check flight mode =====
print("\n" + "=" * 50)
print("  STEP 2: MODE CHECK")
print("=" * 50)
msg = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
if msg:
    mode = mavutil.mode_string_v10(msg)
    armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    print(f"  Mode:  {mode}")
    print(f"  Armed: {armed}")
    if mode != 'STABILIZE':
        print("  -> Switching to STABILIZE for motor test...")
        mode_id = conn.mode_mapping().get('STABILIZE')
        if mode_id is not None:
            conn.set_mode(mode_id)
            time.sleep(1)
            print("  Mode set to STABILIZE")
else:
    print("  No heartbeat received")

input("\nPress Enter for Step 3...")

# ===== STEP 3: Check servo outputs =====
print("\n" + "=" * 50)
print("  STEP 3: SERVO/MOTOR OUTPUT CHECK")
print("=" * 50)
print("  Reading servo outputs (what Cube sends to ESCs)...")
msg = conn.recv_match(type='SERVO_OUTPUT_RAW', blocking=True, timeout=5)
if msg:
    print(f"  Motor 1 (servo1): {msg.servo1_raw} us")
    print(f"  Motor 2 (servo2): {msg.servo2_raw} us")
    print(f"  Motor 3 (servo3): {msg.servo3_raw} us")
    print(f"  Motor 4 (servo4): {msg.servo4_raw} us")
    print()
    print("  Normal values:")
    print("    Disarmed: ~1000 us (no signal)")
    print("    Armed idle: ~1100-1200 us")
    print("    Spinning: >1200 us")
    if all(v < 900 for v in [msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw]):
        print("\n  WARNING: All outputs are 0 - Cube may not have motor outputs configured")
else:
    print("  No servo output data - try requesting it")

input("\nPress Enter for Step 4...")

# ===== STEP 4: Try motor test one at a time =====
print("\n" + "=" * 50)
print("  STEP 4: INDIVIDUAL MOTOR TEST")
print("=" * 50)
print("  Will test Motor 1 at 15% for 3 seconds")
print("  Watch: does the ESC beep? Does the motor twitch?")
input("  PROPS OFF? Press Enter to spin Motor 1...")

conn.mav.command_long_send(
    conn.target_system, conn.target_component,
    mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST, 0,
    1,    # motor 1
    0,    # throttle type = percent
    15,   # 15% throttle
    3,    # 3 seconds
    0, 0, 0
)

# Check for ACK
ack = conn.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if ack:
    result = ack.result
    if result == 0:
        print(f"  Command ACCEPTED by Cube")
    elif result == 4:
        print(f"  Command FAILED - Cube rejected motor test")
        print("  Possible reasons:")
        print("    - Motor test disabled in params (check MOT_SPIN_ARM)")
        print("    - Safety switch not pressed (orange button on GPS module)")
    else:
        print(f"  Command result: {result}")
else:
    print("  No response from Cube")

print("\n  Waiting 3 seconds for motor to finish...")
time.sleep(4)

# Check servo outputs during/after test
msg = conn.recv_match(type='SERVO_OUTPUT_RAW', blocking=True, timeout=3)
if msg:
    print(f"\n  Servo outputs now:")
    print(f"    Motor 1: {msg.servo1_raw} us")
    print(f"    Motor 2: {msg.servo2_raw} us")
    print(f"    Motor 3: {msg.servo3_raw} us")
    print(f"    Motor 4: {msg.servo4_raw} us")

input("\nPress Enter for Step 5...")

# ===== STEP 5: Check for safety switch =====
print("\n" + "=" * 50)
print("  STEP 5: SAFETY SWITCH CHECK")
print("=" * 50)
print("  Many Cube setups have a SAFETY SWITCH (orange button)")
print("  on the GPS module. It must be HELD for 3 seconds to")
print("  enable motor output.")
print()
print("  Look for a small orange/red button on your GPS module.")
print("  If it exists, HOLD it for 3 seconds until it goes solid.")
print("  Then try the motor test again.")
print()
print("  Alternatively, safety switch can be disabled in params:")
print("    BRD_SAFETY_DEFLT = 0  (or BRD_SAFETYENABLE = 0)")

print("\n" + "=" * 50)
print("  SUMMARY")
print("=" * 50)
print("  If motors still don't spin, check:")
print("  1. Power supply voltage (needs 14.8V+ for motors)")
print("  2. Safety switch pressed (orange button on GPS)")
print("  3. ESC signal wires connected to Cube MAIN OUT 1-4")
print("  4. ESC power wires connected to power distribution")
print("  5. Motor wires connected to ESCs (3 wires each)")
print("=" * 50)

conn.close()
