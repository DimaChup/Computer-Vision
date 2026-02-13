# Deploying SAR Mission to Raspberry Pi + Cube Flight Controller

## Your Setup
- Flight Controller: **The Cube (ArduPilot)**
- Companion Computer: **Raspberry Pi** with Pi Camera
- Communication: **MAVLink** protocol (Pi talks to Cube via USB or UART)

---

## STEP 1: Connect to your Pi from your laptop

You need SSH so you can work from your laptop instead of plugging a monitor into the Pi.

### 1a. Find your Pi on the network
Plug the Pi into the same WiFi/ethernet as your laptop, then:

```bash
# On your Windows laptop (PowerShell):
ping raspberrypi.local
```

If that doesn't work, check your router's admin page for the Pi's IP address, or plug a monitor in temporarily.

### 1b. Enable SSH on the Pi
If SSH isn't enabled yet, plug in a monitor+keyboard temporarily:
```bash
sudo raspi-config
# Navigate to: Interface Options -> SSH -> Enable
```

### 1c. SSH in from your laptop
```powershell
# From PowerShell on your Windows laptop:
ssh pi@raspberrypi.local
# Default password is usually: raspberry (CHANGE IT after first login)
```

### CHECK: You can type commands on the Pi from your laptop

---

## STEP 2: Push your code to the Pi

### Option A: Git (recommended)
```bash
# On the Pi:
cd ~
git clone <your-repo-url>
```

### Option B: SCP (copy files directly)
```powershell
# From PowerShell on your laptop:
scp -r "C:\Users\Bristol\Desktop\AI for Robotics\v3\*" pi@raspberrypi.local:~/sar_mission/
```

### CHECK: Your files are on the Pi. SSH in and run `ls ~/sar_mission/`

---

## STEP 3: Install Python dependencies on the Pi

```bash
# On the Pi:
sudo apt update
sudo apt install python3-pip python3-venv

cd ~/sar_mission
python3 -m venv venv
source venv/bin/activate

pip install numpy
pip install dronekit        # MAVLink communication with the Cube
pip install pymavlink       # Low-level MAVLink (installed with dronekit)
pip install picamera2       # Pi Camera control (or picamera for older module)
```

You do NOT need matplotlib on the Pi — there's no screen to show the plot.

### CHECK: Run `python3 -c "import dronekit; print('OK')"` — should print OK

---

## STEP 4: Connect Pi to Cube via USB

Plug a USB cable from the Pi's USB port to the Cube's USB port (the micro-USB on the Cube).

```bash
# On the Pi, check it shows up:
ls /dev/ttyACM*
# Should show: /dev/ttyACM0  (this is the Cube)
```

If you see `/dev/ttyUSB0` instead, that's fine — use that.

### CHECK: The device file exists

---

## STEP 5: Test MAVLink connection (READ ONLY — no flying)

Create a tiny test script on the Pi:

```python
# test_connection.py
from dronekit import connect

print("Connecting to Cube...")
vehicle = connect('/dev/ttyACM0', baud=115200, wait_ready=True)

print(f"Firmware:  {vehicle.version}")
print(f"GPS:       {vehicle.gps_0}")
print(f"Battery:   {vehicle.battery}")
print(f"Altitude:  {vehicle.location.global_relative_frame.alt}")
print(f"Mode:      {vehicle.mode.name}")
print(f"Armed:     {vehicle.armed}")

vehicle.close()
print("Done!")
```

```bash
python3 test_connection.py
```

### CHECK: You see firmware version, GPS info, battery level printed. This means Pi <-> Cube communication works. Nothing moves, nothing arms — this is read-only.

---

## STEP 6: Test the Pi Camera

```python
# test_camera.py
from picamera2 import Picamera2
import time

cam = Picamera2()
cam.configure(cam.create_still_configuration())
cam.start()
time.sleep(2)
cam.capture_file("test_photo.jpg")
cam.stop()
print("Photo saved as test_photo.jpg")
```

```bash
python3 test_camera.py
# Then copy the photo to your laptop to check:
# scp pi@raspberrypi.local:~/sar_mission/test_photo.jpg .
```

### CHECK: Photo looks correct, camera is working

---

## STEP 7: Test arming + takeoff on the GROUND (props OFF!)

IMPORTANT: REMOVE ALL PROPELLERS before this step.

```python
# test_arm.py
from dronekit import connect, VehicleMode
import time

vehicle = connect('/dev/ttyACM0', baud=115200, wait_ready=True)

print(f"Mode: {vehicle.mode.name}")
print(f"Armed: {vehicle.armed}")

# Switch to GUIDED mode (allows programmatic control)
vehicle.mode = VehicleMode("GUIDED")
time.sleep(2)
print(f"Mode now: {vehicle.mode.name}")

# Try to arm (will only work if pre-arm checks pass)
vehicle.armed = True
time.sleep(3)
print(f"Armed: {vehicle.armed}")

# Disarm immediately
vehicle.armed = False
time.sleep(2)
print(f"Armed after disarm: {vehicle.armed}")

vehicle.close()
print("Done!")
```

### CHECK: Drone arms and disarms. Motors may briefly twitch — that's normal. NO PROPS ON.

---

## STEP 8: Understanding the code adaptation

Your simple_simulator.py has two parts:
1. **State machine logic** (the brain) — THIS STAYS
2. **Matplotlib visualization** (the eyes) — THIS GOES

What replaces what:
| Simulator (fake)              | Real drone (Pi + Cube)                    |
|-------------------------------|-------------------------------------------|
| `self.drone.x, self.drone.y` | `vehicle.location.global_relative_frame`  |
| `self.move_towards(target)`   | `vehicle.simple_goto(LocationGlobal(...))`|
| `self.drone.altitude += 1.0`  | `vehicle.simple_takeoff(30)`              |
| `self.check_detection()` (FOV math) | Camera + ML model (TFLite/YOLO)   |
| `plt.pause(0.01)`             | `time.sleep(0.1)`                         |
| `on_key_press` (matplotlib)   | MAVLink messages or ground station comms  |
| `matplotlib` plots             | Nothing (or send telemetry to laptop)     |

---

## STEP 9: Gradual testing order

Test each phase separately before combining:

1. **Takeoff + land** — GUIDED mode, takeoff to 5m, hover 10 sec, land
2. **Waypoint following** — Fly to 2-3 GPS waypoints and return
3. **Camera detection** — While hovering, test if camera can detect target
4. **Single state transitions** — Test TAKEOFF -> TRANSIT -> hover at first waypoint -> RTL
5. **Full mission** — Run complete state machine

NEVER skip steps. ALWAYS test with low altitude first (5m).

---

## Key DroneKit commands you'll need

```python
from dronekit import connect, VehicleMode, LocationGlobalRelative

# Connect
vehicle = connect('/dev/ttyACM0', baud=115200, wait_ready=True)

# Takeoff
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True
vehicle.simple_takeoff(30)  # 30 metres

# Fly to GPS coordinate
target = LocationGlobalRelative(51.4545, -2.6030, 30)  # lat, lon, alt
vehicle.simple_goto(target)

# Check position
lat = vehicle.location.global_relative_frame.lat
lon = vehicle.location.global_relative_frame.lon
alt = vehicle.location.global_relative_frame.alt

# Land
vehicle.mode = VehicleMode("LAND")

# Return to launch
vehicle.mode = VehicleMode("RTL")
```
