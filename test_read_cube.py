"""
Read live data from the Cube. READ ONLY - nothing moves.
Press Ctrl+C to stop.
Run on Pi: python3 test_read_cube.py
"""
from pymavlink import mavutil
import time
import os

print("Connecting to Cube...")
conn = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600)
conn.wait_heartbeat(timeout=10)
print("Connected!")

# Request all data streams at 4Hz
print("Requesting data streams...")
conn.mav.request_data_stream_send(
    conn.target_system, conn.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1
)
time.sleep(1)
print("Streams requested. Press Ctrl+C to stop.\n")

# Store latest values
data = {
    'mode': '?', 'armed': False,
    'lat': 0, 'lon': 0, 'alt': 0, 'sats': 0, 'fix': 0,
    'voltage': 0, 'current': 0, 'remaining': 0,
    'roll': 0, 'pitch': 0, 'yaw': 0,
    'groundspeed': 0, 'heading': 0, 'throttle': 0,
    'alt_agl': 0,
}

try:
    while True:
        msg = conn.recv_match(blocking=True, timeout=1)
        if not msg:
            continue
        t = msg.get_type()
        if t == 'BAD_DATA':
            continue

        if t == 'HEARTBEAT':
            data['mode'] = mavutil.mode_string_v10(msg)
            data['armed'] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == 'GPS_RAW_INT':
            data['lat'] = msg.lat / 1e7
            data['lon'] = msg.lon / 1e7
            data['alt'] = msg.alt / 1000
            data['sats'] = msg.satellites_visible
            data['fix'] = msg.fix_type
        elif t == 'SYS_STATUS':
            data['voltage'] = msg.voltage_battery / 1000
            data['current'] = msg.current_battery / 100
            data['remaining'] = msg.battery_remaining
        elif t == 'ATTITUDE':
            data['roll'] = msg.roll * 57.3
            data['pitch'] = msg.pitch * 57.3
            data['yaw'] = msg.yaw * 57.3
        elif t == 'VFR_HUD':
            data['groundspeed'] = msg.groundspeed
            data['heading'] = msg.heading
            data['throttle'] = msg.throttle
        elif t == 'GLOBAL_POSITION_INT':
            data['alt_agl'] = msg.relative_alt / 1000

        # Clear screen and print dashboard
        os.system('clear')
        print("=" * 45)
        print("  CUBE LIVE DATA  (Ctrl+C to stop)")
        print("=" * 45)
        print(f"  Mode:        {data['mode']}")
        print(f"  Armed:       {data['armed']}")
        print()
        print(f"  GPS Lat:     {data['lat']:.7f}")
        print(f"  GPS Lon:     {data['lon']:.7f}")
        print(f"  Satellites:  {data['sats']}")
        print(f"  Fix type:    {data['fix']} (3=3D fix)")
        print()
        print(f"  Altitude:    {data['alt']:.1f} m (MSL)")
        print(f"  Alt AGL:     {data['alt_agl']:.1f} m")
        print(f"  Groundspeed: {data['groundspeed']:.1f} m/s")
        print(f"  Heading:     {data['heading']} deg")
        print(f"  Throttle:    {data['throttle']}%")
        print()
        print(f"  Roll:        {data['roll']:.1f} deg")
        print(f"  Pitch:       {data['pitch']:.1f} deg")
        print(f"  Yaw:         {data['yaw']:.1f} deg")
        print()
        print(f"  Battery:     {data['voltage']:.2f}V  {data['current']:.1f}A  {data['remaining']}%")
        print("=" * 45)

except KeyboardInterrupt:
    print("\nStopped.")
    conn.close()
