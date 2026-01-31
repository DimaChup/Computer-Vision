# filename: vision.py
import cv2
import numpy as np

class VisionSystem:
    def __init__(self, camera_index=0):
        self.cap = None
        # Only open camera if index is provided (Real Flight Mode)
        if camera_index is not None:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1456)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1088)
        
        # --- TUNING: RED COLOR DETECTION (HSV) ---
        # Red wraps around 0/180, so we need two ranges
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.min_area = 200 # Minimum blob size

    def detect_in_image(self, frame):
        """ Core logic used by both Real Drone and Simulator """
        if frame is None: return False, 0, 0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Combine both red masks
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        full_mask = mask1 + mask2

        # Clean noise
        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

        # Find Blobs
        contours, _ = cv2.findContours(full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_area:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return True, cx, cy

        return False, 0, 0

    def get_latest_detection(self):
        """ For REAL DRONE: Reads from camera """
        if not self.cap: return False, 0, 0
        ret, frame = self.cap.read()
        if not ret: return False, 0, 0
        return self.detect_in_image(frame)

    def process_frame_manually(self, frame):
        """ For SIMULATOR: Processes injected image """
        return self.detect_in_image(frame)
        
    def release(self):
        if self.cap: self.cap.release()