# filename: data_capture.py
import cv2
import time
import os
from datetime import datetime

SAVE_DIR = "training_data"
INTERVAL = 1.0 

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # Open Camera (Real)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1456)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1088)

    print(f"Capturing to {SAVE_DIR} every {INTERVAL}s. Press CTRL+C to stop.")
    last_shot = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            if time.time() - last_shot > INTERVAL:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{SAVE_DIR}/img_{ts}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
                last_shot = time.time()
                
    except KeyboardInterrupt:
        print("Stopped.")
        cap.release()

if __name__ == "__main__":
    main()