# config.py
# ==========================================
#       CONFIGURATION & SETTINGS
# ==========================================
import sys

# --- OPERATION MODE (auto-detect platform) ---
# Windows = SIMULATION (laptop dev), Linux = REAL (Pi deployment)
if sys.platform == "win32":
    MODE = "SIMULATION"
else:
    MODE = "REAL"

# --- FLIGHT CONNECTION ---
if MODE == "REAL":
    CONNECTION_STR = '/dev/ttyAMA0'
    BAUD_RATE = 921600
else:
    CONNECTION_STR = 'tcp:127.0.0.1:5762'
    BAUD_RATE = 57600

# --- ALTITUDES ---
TARGET_ALT = 30.0 # Search Altitude (Meters)
VERIFY_ALT = 15.0 # Descent Altitude for Verification

# --- MAP CONFIGURATION (Simulation Only) ---
MAP_FILE = "map.jpg"
DUMMY_FILE = "dummy.png"
MAP_WIDTH_METERS = 480.0  
REF_LAT = 51.425106  
REF_LON = -2.672257

# --- TARGET SPECS ---
TARGET_REAL_RADIUS_M = 0.15 # 15 cm radius 
DUMMY_HEIGHT_M = 1.8        

# --- CAMERA SPECS ---
# Update these for the Raspberry Pi Global Shutter Camera
SENSOR_WIDTH_MM = 5.02
FOCAL_LENGTH_MM = 6.0
IMAGE_W = 640
IMAGE_H = 480
REAL_CAMERA_INDEX = 0 # Usually 0 for Pi Cam

# --- SPEED SETTINGS ---
TRANSIT_SPEED_MPS = 15.0  
SEARCH_SPEED_MPS = 10.0   

# --- LOGGING ---
LOG_FILE = "flight_log.csv"