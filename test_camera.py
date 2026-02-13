"""
Test Pi Camera - no Qt/display needed.
Captures a photo and prints camera info.
Run on Pi: python3 test_camera.py
"""
from picamera2 import Picamera2
import time

print("Starting camera test...")

cam = Picamera2()
config = cam.create_still_configuration(main={"size": (640, 480)})
cam.configure(config)
cam.start()
print("Camera started. Warming up...")
time.sleep(2)

# Capture a test photo
cam.capture_file("test_photo.jpg")
print("Photo saved as test_photo.jpg")

# Grab a frame as numpy array (this is what main.py will use)
frame = cam.capture_array()
print(f"Frame shape: {frame.shape}  dtype: {frame.dtype}")

cam.stop()
print("Camera test PASSED!")
