# vision.py
import cv2
import numpy as np
import os
import sys
import time

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
        # Real Camera Setup
        if camera_index is not None:
            print(f"[VISION] Opening Camera Index {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            # Set to standard 640x480 for speed, or higher if needed
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # --- AI MODEL SETUP ---
        self.model = None
        self.using_ai = False
        
        # Pre-check dependencies
        tflite_ready = True
        if model_path.endswith('.tflite'):
            try: import tensorflow
            except ImportError: 
                try: import tflite_runtime
                except ImportError: tflite_ready = False

        if AI_AVAILABLE and tflite_ready and os.path.exists(model_path):
            try:
                print(f"[VISION] Loading Model: {model_path}...")
                self.model = YOLO(model_path, task='detect')
                
                # Warmup run
                print("[VISION] Verifying AI Engine...")
                self.model(np.zeros((100, 100, 3), dtype=np.uint8), verbose=False)
                
                self.using_ai = True
                print("[VISION] AI Engine Loaded Successfully!")
            except Exception as e:
                print(f"[VISION] AI Load Failed: {e}")
                self.using_ai = False
        else:
            print("[VISION] AI Not Available. Using basic logic.")

    def detect_in_image(self, frame):
        """ Returns: found (bool), x, y, confidence (float) """
        if frame is None: return False, 0, 0, 0.0

        # --- AI DETECTION ---
        if self.using_ai:
            results = self.model(frame, conf=0.4, verbose=False)
            if results[0].boxes:
                best_box = max(results[0].boxes, key=lambda x: x.conf[0])
                x, y, w, h = best_box.xywh[0].cpu().numpy()
                conf = float(best_box.conf[0])
                
                # Draw box on frame
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"AI {conf:.2f}", (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                return True, int(x), int(y), conf

        return False, 0, 0, 0.0
    
    def get_frame(self):
        """ Returns raw frame from Real Camera """
        if not self.cap: return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def process_frame_manually(self, frame):
        """ Unified processing for both Sim and Real """
        return self.detect_in_image(frame)

    def release(self):
        if self.cap: self.cap.release()