# vision.py
import cv2
import numpy as np
import os
import sys

try:
    from ultralytics import YOLO
    AI_AVAILABLE = True
except ImportError:
    print("WARNING: 'ultralytics' not found. AI features disabled.")
    AI_AVAILABLE = False

class VisionSystem:
    def __init__(self, camera_index=0, model_path="best.tflite"):
        self.cap = None
        # Real Camera Initialization
        if camera_index is not None:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.model = None
        self.using_ai = False
        
        if AI_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path, task='detect')
                # Warmup
                self.model(np.zeros((100,100,3), dtype=np.uint8), verbose=False)
                self.using_ai = True
                print("[VISION] AI Engine Ready.")
            except Exception as e:
                print(f"[VISION] Model load failed: {e}")
                
    def get_frame(self):
        """ Returns the raw frame from camera (or None) """
        if not self.cap: return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def detect_in_image(self, frame):
        """ Process an image. Returns (found, x, y, conf) """
        if frame is None: return False, 0, 0, 0.0

        if self.using_ai:
            results = self.model(frame, conf=0.4, verbose=False)
            if results[0].boxes:
                best_box = max(results[0].boxes, key=lambda x: x.conf[0])
                x, y, w, h = best_box.xywh[0].cpu().numpy()
                conf = float(best_box.conf[0])
                return True, int(x), int(y), conf
        return False, 0, 0, 0.0

    def process_frame_manually(self, frame):
        return self.detect_in_image(frame)
        
    def get_latest_detection(self):
        """ Captures and processes a frame in one go """
        frame = self.get_frame()
        return self.detect_in_image(frame)

    def release(self):
        if self.cap: self.cap.release()