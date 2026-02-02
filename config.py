# config.py
# ==========================================
#       CALIBRATION & CONFIGURATION
# ==========================================

CONNECTION_STR = 'tcp:127.0.0.1:5762'
TARGET_ALT = 40.0 # Search Altitude
VERIFY_ALT = 15.0 # Descent Altitude for Verification

# 1. Map Configuration
MAP_FILE = "map.jpg"
DUMMY_FILE = "dummy.png"
MAP_WIDTH_METERS = 480.0  # Total width of the map image in meters
REF_LAT = 51.425106  
REF_LON = -2.672257

# 2. Target Size Configuration
TARGET_REAL_RADIUS_M = 0.15 # 15 cm radius for the red dot
DUMMY_HEIGHT_M = 1.8        # 6 ft (1.8m) tall dummy

# 3. Camera Specs
SENSOR_WIDTH_MM = 5.02
FOCAL_LENGTH_MM = 6.0
IMAGE_W = 640
IMAGE_H = 480

# 4. Safety
NFZ_BUFFER_M = 10.0 # Meters to keep away from NFZ

# 5. Speed Settings
TRANSIT_SPEED_MPS = 15.0  # Speed when flying TO the search area (m/s)
SEARCH_SPEED_MPS = 10.0   # Speed while scanning the grid (m/s)