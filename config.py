# config.py
# ==========================================
#       CONFIGURATION & SETTINGS
# ==========================================

# --- OPERATION MODE ---
# Set to "SIMULATION" for laptop testing with map.jpg
# Set to "REAL" for flying the actual drone with a camera
MODE = "SIMULATION" 

# --- FLIGHT CONNECTION ---
# Sim: 'tcp:127.0.0.1:5762'
# Real (Pi to Cube via Serial): '/dev/ttyACM0' or '/dev/ttyAMA0'
CONNECTION_STR = 'tcp:127.0.0.1:5762'
BAUD_RATE = 57600

# --- ALTITUDES ---
TARGET_ALT = 40.0 # Search Altitude (Meters)
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
# Important: Update these for the Raspberry Pi Global Shutter Camera
SENSOR_WIDTH_MM = 5.02
FOCAL_LENGTH_MM = 6.0
IMAGE_W = 640
IMAGE_H = 480
REAL_CAMERA_INDEX = 0 # 0 is usually the default Pi Camera

# --- SAFETY ---
NFZ_BUFFER_M = 10.0 # Buffer distance from No-Fly Zone

# --- SPEED SETTINGS ---
TRANSIT_SPEED_MPS = 15.0  
SEARCH_SPEED_MPS = 10.0   

# --- LOGGING ---
LOG_FILE = "flight_log.csv"