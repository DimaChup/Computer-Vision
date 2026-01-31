import cv2
import numpy as np
import os
import random

# --- CONFIG ---
BACKGROUND_IMG = "map.jpg"  # Your grass map
FOREGROUND_IMG = "dummy.png" # Your transparent dummy
OUTPUT_DIR = "dataset"
NUM_IMAGES = 200            # How many training images to generate?
IMG_SIZE = 640              # YOLO standard input size

def create_synthetic_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(f"{OUTPUT_DIR}/images")
        os.makedirs(f"{OUTPUT_DIR}/labels")

    # Load Assets
    bg_full = cv2.imread(BACKGROUND_IMG)
    # Read FG with Alpha channel (Transparency)
    fg_full = cv2.imread(FOREGROUND_IMG, cv2.IMREAD_UNCHANGED)
    
    if bg_full is None or fg_full is None:
        print("Error: Missing map.jpg or dummy.png")
        return

    print(f"Generating {NUM_IMAGES} synthetic training images...")

    for i in range(NUM_IMAGES):
        # 1. Create Random Background Crop
        h_bg, w_bg = bg_full.shape[:2]
        x_rnd = random.randint(0, w_bg - IMG_SIZE)
        y_rnd = random.randint(0, h_bg - IMG_SIZE)
        background = bg_full[y_rnd:y_rnd+IMG_SIZE, x_rnd:x_rnd+IMG_SIZE].copy()

        # 2. Randomize Dummy (Scale & Rotation)
        # Scale: Simulate altitude (small = high, large = low)
        scale = random.uniform(0.1, 0.4) 
        h_fg, w_fg = fg_full.shape[:2]
        new_w, new_h = int(w_fg * scale), int(h_fg * scale)
        
        # Resize
        fg_resized = cv2.resize(fg_full, (new_w, new_h))
        
        # Rotate
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((new_w//2, new_h//2), angle, 1.0)
        fg_rotated = cv2.warpAffine(fg_resized, M, (new_w, new_h))

        # 3. Paste Dummy onto Background
        # Random position on the 640x640 canvas
        paste_x = random.randint(0, IMG_SIZE - new_w)
        paste_y = random.randint(0, IMG_SIZE - new_h)

        # Alpha Blending Logic
        alpha_s = fg_rotated[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            background[paste_y:paste_y+new_h, paste_x:paste_x+new_w, c] = \
                (alpha_s * fg_rotated[:, :, c] +
                 alpha_l * background[paste_y:paste_y+new_h, paste_x:paste_x+new_w, c])

        # 4. Calculate Bounding Box (YOLO Format)
        # Format: <class_id> <x_center> <y_center> <width> <height> (Normalized 0-1)
        x_center = (paste_x + new_w / 2) / IMG_SIZE
        y_center = (paste_y + new_h / 2) / IMG_SIZE
        bbox_w = new_w / IMG_SIZE
        bbox_h = new_h / IMG_SIZE

        # 5. Save
        filename = f"sim_dummy_{i:04d}"
        
        # Save Image
        cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}.jpg", background)
        
        # Save Label
        with open(f"{OUTPUT_DIR}/labels/{filename}.txt", "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")
        
        print(f"Generated {filename}")

if __name__ == "__main__":
    create_synthetic_data()