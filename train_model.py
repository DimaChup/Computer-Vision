# filename: train_model.py
from ultralytics import YOLO
import os

def main():
    print("--- YOLOv8 TRAINING SCRIPT ---")
    
    # 1. Load the Model
    # We use 'yolov8n.pt' (Nano). It is the smallest/fastest version.
    # Perfect for Raspberry Pi 5.
    print("Loading YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt") 

    # 2. Train the Model
    # epochs=50: The AI will look at your entire dataset 50 times.
    # imgsz=640: The resolution it learns at.
    # device=0: Uses GPU if available (auto-detects)
    print("Starting Training... (This takes 15-30 mins on GPU)")
    results = model.train(
        data="data.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16,
        name="dummy_detector" # The name of the output folder
    )
    
    print("Training Complete!")

    # 3. Export to TFLite (for Raspberry Pi)
    # The Pi runs 'TensorFlow Lite' files much faster than standard files.
    # int8=True simplifies the math to make it run 3x faster without losing much accuracy.
    print("Exporting to TFLite format...")
    try:
        success = model.export(format="tflite", int8=True)
        if success:
            print(f"SUCCESS! Your new brain is ready.")
            print(f"Exported file: {success}")
            print("Look for the file ending in '.tflite' inside the 'runs' folder.")
    except Exception as e:
        print(f"Export failed: {e}")
        print("Note: Exporting to TFLite often requires running on Linux/Colab.")

if __name__ == "__main__":
    # Simple check to ensure data config exists
    if not os.path.exists("data.yaml"):
        print("ERROR: data.yaml not found!")
    else:
        main()