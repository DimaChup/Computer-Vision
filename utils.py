# utils.py
import math
import cv2
import config

class GeoTransformer:
    def __init__(self, map_w_px):
        self.pix_per_m = map_w_px / config.MAP_WIDTH_METERS

    def gps_to_pixels(self, lat, lon):
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(lat))
        lon_m = 111132.954 * math.cos(math.radians(lat))
        dy = -(lat - config.REF_LAT) * lat_m
        dx = (lon - config.REF_LON) * lon_m
        return int(dx * self.pix_per_m), int(dy * self.pix_per_m)
        
    def pixels_to_gps(self, x, y):
        dx = x / self.pix_per_m
        dy = y / self.pix_per_m
        lat_m = 111132.954 - 559.822 * math.cos(2 * math.radians(config.REF_LAT))
        lon_m = 111132.954 * math.cos(math.radians(config.REF_LAT))
        dLat = -(dy / lat_m)
        dLon = dx / lon_m
        return config.REF_LAT + dLat, config.REF_LON + dLon

def overlay_image_alpha(background, overlay, x, y, target_w, target_h, rotation_deg=0):
    if overlay is None: return
    h_src, w_src = overlay.shape[:2]
    if target_w == 0:
        scale = target_h / h_src
        target_w = int(w_src * scale)
    if target_w <= 0 or target_h <= 0: return
    try: resized = cv2.resize(overlay, (target_w, target_h))
    except: return 
    
    diag = int(math.sqrt(target_w**2 + target_h**2))
    pad_x = (diag - target_w) // 2
    pad_y = (diag - target_h) // 2
    padded = cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0,0,0,0))
    h_pad, w_pad = padded.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w_pad//2, h_pad//2), rotation_deg, 1.0)
    rotated = cv2.warpAffine(padded, M_rot, (w_pad, h_pad))
    
    y1 = y - h_pad // 2; y2 = y1 + h_pad
    x1 = x - w_pad // 2; x2 = x1 + w_pad
    h_bg, w_bg = background.shape[:2]
    y1_c = max(0, y1); y2_c = min(h_bg, y2)
    x1_c = max(0, x1); x2_c = min(w_bg, x2)
    if y1_c >= y2_c or x1_c >= x2_c: return
    ov_y1 = y1_c - y1; ov_y2 = ov_y1 + (y2_c - y1_c)
    ov_x1 = x1_c - x1; ov_x2 = ov_x1 + (x2_c - x1_c)
    overlay_crop = rotated[ov_y1:ov_y2, ov_x1:ov_x2]
    bg_crop = background[y1_c:y2_c, x1_c:x2_c]
    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        for c in range(0, 3):
            bg_crop[:, :, c] = (1. - alpha) * bg_crop[:, :, c] + alpha * overlay_crop[:, :, c]
    else:
        background[y1_c:y2_c, x1_c:x2_c] = overlay_crop