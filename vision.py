import cv2
import numpy as np
import os
import sys

# Try to import Ultralytics (YOLO)
try:
    from ultralytics import YOLO
    AI_AVAILABLE = True
except ImportError:
    print("WARNING: 'ultralytics' library not found. Run 'pip install ultralytics'")
    AI_AVAILABLE = False

class VisionSystem:
    def __init__(self, camera_index=0, model_path="best.tflite"):
        self.cap = None
        # Only open camera if index is provided (Real Flight Mode)
        if camera_index is not None:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1456)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1088)
        
        # --- AI MODEL SETUP ---
        self.model = None
        self.using_ai = False
        
        if AI_AVAILABLE and os.path.exists(model_path):
            print(f"[VISION] Loading AI Model: {model_path}...")
            try:
                # Load TFLite model
                self.model = YOLO(model_path, task='detect')
                
                # CRITICAL FIX: Run a dummy inference immediately to check dependencies.
                # YOLOv8 lazy-loads the backend (TensorFlow), so it usually crashes 
                # later during the flight. We force it to crash HERE if dependencies are missing.
                print("[VISION] Verifying backend libraries (TensorFlow)...")
                dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                self.model(dummy_frame, verbose=False)
                
                self.using_ai = True
                print("[VISION] AI Model Verified & Loaded Successfully!")
                
            except Exception as e:
                # This catches 'ModuleNotFoundError: No module named tensorflow'
                print(f"\n[VISION] AI LOAD FAILED: {e}")
                print("[VISION] TIP: To use .tflite, you must run: 'pip install tensorflow'")
                print("[VISION] AUTOMATICALLY FALLING BACK TO RED DOT DETECTION.\n")
                self.using_ai = False
                self.model = None
        else:
            if not os.path.exists(model_path):
                print(f"[VISION] '{model_path}' not found.")
            print("[VISION] Falling back to Color Detection.")

        # --- FALLBACK: RED COLOR TUNING ---
        # (Used only if AI fails to load)
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])
        self.min_area = 200 

    def detect_in_image(self, frame):
        """ 
        Main detection function. 
        Returns: found (bool), center_x (int), center_y (int)
        """
        if frame is None: return False, 0, 0

        # --- STRATEGY A: AI DETECTION (YOLO) ---
        if self.using_ai:
            # Run inference
            # conf=0.4: Only accept detections with >40% confidence
            results = self.model(frame, conf=0.4, verbose=False)
            
            if results[0].boxes:
                # Find the detection with the highest confidence
                best_box = max(results[0].boxes, key=lambda x: x.conf[0])
                
                # YOLO returns center_x, center_y, width, height
                x, y, w, h = best_box.xywh[0].cpu().numpy()
                
                # Draw a box on the frame for debugging (visible in dashboard)
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"DUMMY {best_box.conf[0]:.2f}", (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                return True, int(x), int(y)
            else:
                return False, 0, 0

        # --- STRATEGY B: RED COLOR (FALLBACK) ---
        # Only runs if AI is not loaded
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        full_mask = mask1 + mask2
        
        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
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