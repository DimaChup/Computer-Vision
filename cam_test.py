"""
Simple camera test - no display needed.
Run on Pi: python3 cam_test.py
"""
from picamera2 import Picamera2
import time

cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"size": (640, 480)}))
cam.start()
time.sleep(2)

cam.capture_file("photo.jpg")
frame = cam.capture_array()
print(f"Camera works! Frame: {frame.shape}")

cam.stop()
