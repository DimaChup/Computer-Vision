"""
Test 1: Pi Camera Feed
Run this on the Pi to check if the camera works.
It will show a live preview window for 10 seconds, then save a photo.
"""
from picamera2 import Picamera2
from picamera2.previews.qt import QGlPicamera2
import time

print("Starting camera test...")

cam = Picamera2()

# Show live preview (you'll see it on the Pi's screen)
cam.configure(cam.create_preview_configuration())
cam.start_preview(True)  # True = show on screen
cam.start()

print("Camera preview is live! Showing for 10 seconds...")
time.sleep(10)

# Save a test photo
cam.switch_mode_and_capture_file(cam.create_still_configuration(), "test_photo.jpg")
print("Photo saved as test_photo.jpg")

cam.stop_preview()
cam.stop()
print("Camera test PASSED!")
