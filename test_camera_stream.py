"""
Live camera stream using picamera2 preview.
Run on Pi: python3 test_camera_stream.py
Press Ctrl+C to stop.
"""
from picamera2 import Picamera2, Preview
import time

print("Starting live camera stream...")

cam = Picamera2()
config = cam.create_preview_configuration(main={"size": (640, 480)})
cam.configure(config)

# Try DRM preview first (works without desktop), fallback to Qt
try:
    cam.start_preview(Preview.DRM)
    print("Using DRM preview (direct to screen)")
except Exception:
    try:
        cam.start_preview(Preview.QTGL)
        print("Using Qt preview")
    except Exception:
        cam.start_preview(Preview.NULL)
        print("No preview available - but camera is running")

cam.start()
print("Live stream running! Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping...")

cam.stop_preview()
cam.stop()
print("Done.")
