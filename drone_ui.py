"""
Simple Drone Control UI - runs on Pi with screen.
Shows live data + basic safe commands.
Run on Pi: python3 drone_ui.py
"""
import tkinter as tk
from tkinter import messagebox
from pymavlink import mavutil
import threading
import time

# Connection settings
PORT = '/dev/ttyAMA0'
BAUD = 921600


class DroneUI:
    def __init__(self):
        self.conn = None
        self.running = True
        self.data = {
            'mode': '?', 'armed': False,
            'lat': 0.0, 'lon': 0.0, 'alt': 0.0, 'alt_agl': 0.0,
            'sats': 0, 'fix': 0,
            'voltage': 0.0, 'current': 0.0, 'remaining': 0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'groundspeed': 0.0, 'heading': 0, 'throttle': 0,
        }

        # Connect to Cube
        print("Connecting to Cube...")
        self.conn = mavutil.mavlink_connection(PORT, baud=BAUD)
        self.conn.wait_heartbeat(timeout=10)
        print("Connected!")

        # Request data streams
        self.conn.mav.request_data_stream_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1
        )

        # Build UI
        self.root = tk.Tk()
        self.root.title("SAR Drone Control")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("500x700")

        # Title
        tk.Label(self.root, text="SAR DRONE", font=("Helvetica", 20, "bold"),
                 bg='#1a1a2e', fg='#22b8cf').pack(pady=10)

        # Status frame
        status_frame = tk.Frame(self.root, bg='#16213e', padx=15, pady=10)
        status_frame.pack(fill='x', padx=10, pady=5)

        self.mode_label = self._label(status_frame, "Mode: ?", '#ffd43b', 16)
        self.armed_label = self._label(status_frame, "Armed: False", '#e0e0e0', 14)

        # GPS frame
        gps_frame = tk.LabelFrame(self.root, text=" GPS ", font=("Helvetica", 11, "bold"),
                                   bg='#16213e', fg='#4dabf7', padx=10, pady=5)
        gps_frame.pack(fill='x', padx=10, pady=5)

        self.lat_label = self._label(gps_frame, "Lat: 0.0", '#e0e0e0', 11)
        self.lon_label = self._label(gps_frame, "Lon: 0.0", '#e0e0e0', 11)
        self.sats_label = self._label(gps_frame, "Sats: 0  Fix: 0", '#e0e0e0', 11)

        # Flight frame
        flight_frame = tk.LabelFrame(self.root, text=" Flight ", font=("Helvetica", 11, "bold"),
                                      bg='#16213e', fg='#4dabf7', padx=10, pady=5)
        flight_frame.pack(fill='x', padx=10, pady=5)

        self.alt_label = self._label(flight_frame, "Alt: 0.0m", '#e0e0e0', 11)
        self.speed_label = self._label(flight_frame, "Speed: 0.0 m/s  Hdg: 0", '#e0e0e0', 11)
        self.att_label = self._label(flight_frame, "R: 0.0  P: 0.0  Y: 0.0", '#e0e0e0', 11)

        # Battery frame
        batt_frame = tk.LabelFrame(self.root, text=" Battery ", font=("Helvetica", 11, "bold"),
                                    bg='#16213e', fg='#4dabf7', padx=10, pady=5)
        batt_frame.pack(fill='x', padx=10, pady=5)

        self.batt_label = self._label(batt_frame, "0.0V  0.0A  0%", '#e0e0e0', 13)

        # Commands frame
        cmd_frame = tk.LabelFrame(self.root, text=" Commands ", font=("Helvetica", 11, "bold"),
                                   bg='#16213e', fg='#51cf66', padx=10, pady=10)
        cmd_frame.pack(fill='x', padx=10, pady=10)

        # Row 1: Mode buttons
        row1 = tk.Frame(cmd_frame, bg='#16213e')
        row1.pack(fill='x', pady=3)
        self._button(row1, "STABILIZE", lambda: self.set_mode("STABILIZE"), '#4dabf7')
        self._button(row1, "GUIDED", lambda: self.set_mode("GUIDED"), '#ffa94d')
        self._button(row1, "LOITER", lambda: self.set_mode("LOITER"), '#4dabf7')
        self._button(row1, "RTL", lambda: self.set_mode("RTL"), '#e03131')

        # Row 2: Arm / Disarm
        row2 = tk.Frame(cmd_frame, bg='#16213e')
        row2.pack(fill='x', pady=3)
        self._button(row2, "ARM", self.arm, '#51cf66')
        self._button(row2, "DISARM", self.disarm, '#e03131')

        # Row 3: Takeoff / Land
        row3 = tk.Frame(cmd_frame, bg='#16213e')
        row3.pack(fill='x', pady=3)
        self._button(row3, "TAKEOFF 5m", lambda: self.takeoff(5), '#ffd43b')
        self._button(row3, "LAND", lambda: self.set_mode("LAND"), '#e03131')

        # Log area
        log_frame = tk.LabelFrame(self.root, text=" Log ", font=("Helvetica", 10, "bold"),
                                   bg='#16213e', fg='#4dabf7', padx=5, pady=5)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=6, bg='#0f3460', fg='#e0e0e0',
                                font=("Courier", 9), state='disabled')
        self.log_text.pack(fill='both', expand=True)

        self.log("Connected to Cube")

        # Start data reading thread
        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

        # Start UI update loop
        self.update_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def _label(self, parent, text, color, size):
        label = tk.Label(parent, text=text, font=("Courier", size),
                         bg=parent.cget('bg'), fg=color, anchor='w')
        label.pack(fill='x')
        return label

    def _button(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, command=command,
                        bg=color, fg='#1a1a2e', font=("Helvetica", 11, "bold"),
                        width=10, activebackground=color)
        btn.pack(side='left', padx=3, expand=True)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"{time.strftime('%H:%M:%S')} {message}\n")
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def read_loop(self):
        while self.running:
            try:
                msg = self.conn.recv_match(blocking=True, timeout=1)
                if not msg:
                    continue
                t = msg.get_type()
                if t == 'BAD_DATA':
                    continue
                if t == 'HEARTBEAT':
                    self.data['mode'] = mavutil.mode_string_v10(msg)
                    self.data['armed'] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                elif t == 'GPS_RAW_INT':
                    self.data['lat'] = msg.lat / 1e7
                    self.data['lon'] = msg.lon / 1e7
                    self.data['alt'] = msg.alt / 1000
                    self.data['sats'] = msg.satellites_visible
                    self.data['fix'] = msg.fix_type
                elif t == 'SYS_STATUS':
                    self.data['voltage'] = msg.voltage_battery / 1000
                    self.data['current'] = msg.current_battery / 100
                    self.data['remaining'] = msg.battery_remaining
                elif t == 'ATTITUDE':
                    self.data['roll'] = msg.roll * 57.3
                    self.data['pitch'] = msg.pitch * 57.3
                    self.data['yaw'] = msg.yaw * 57.3
                elif t == 'VFR_HUD':
                    self.data['groundspeed'] = msg.groundspeed
                    self.data['heading'] = msg.heading
                    self.data['throttle'] = msg.throttle
                elif t == 'GLOBAL_POSITION_INT':
                    self.data['alt_agl'] = msg.relative_alt / 1000
                elif t == 'STATUSTEXT':
                    self.log(f"FC: {msg.text}")
            except Exception as e:
                self.log(f"Read error: {e}")

    def update_ui(self):
        if not self.running:
            return
        d = self.data
        self.mode_label.config(text=f"Mode: {d['mode']}")
        armed_color = '#e03131' if d['armed'] else '#51cf66'
        self.armed_label.config(text=f"Armed: {d['armed']}", fg=armed_color)
        self.lat_label.config(text=f"Lat: {d['lat']:.7f}")
        self.lon_label.config(text=f"Lon: {d['lon']:.7f}")
        self.sats_label.config(text=f"Sats: {d['sats']}  Fix: {d['fix']} (3=3D)")
        self.alt_label.config(text=f"Alt MSL: {d['alt']:.1f}m  AGL: {d['alt_agl']:.1f}m")
        self.speed_label.config(text=f"Speed: {d['groundspeed']:.1f} m/s  Hdg: {d['heading']}  Thr: {d['throttle']}%")
        self.att_label.config(text=f"R: {d['roll']:.1f}  P: {d['pitch']:.1f}  Y: {d['yaw']:.1f}")

        batt_color = '#e03131' if d['voltage'] < 10.5 and d['voltage'] > 0 else '#51cf66'
        self.batt_label.config(text=f"{d['voltage']:.1f}V  {d['current']:.1f}A  {d['remaining']}%",
                               fg=batt_color)

        self.root.after(250, self.update_ui)

    def set_mode(self, mode_name):
        mode_id = self.conn.mode_mapping().get(mode_name)
        if mode_id is None:
            self.log(f"Unknown mode: {mode_name}")
            return
        self.conn.set_mode(mode_id)
        self.log(f"Mode -> {mode_name}")

    def arm(self):
        if not messagebox.askyesno("ARM", "ARM the drone?\n\nMAKE SURE PROPS ARE OFF FOR TESTING!"):
            return
        self.conn.arducopter_arm()
        self.log("ARM command sent")

    def disarm(self):
        self.conn.arducopter_disarm()
        self.log("DISARM command sent")

    def takeoff(self, alt):
        if not messagebox.askyesno("TAKEOFF", f"TAKEOFF to {alt}m?\n\nDrone must be ARMED and in GUIDED mode."):
            return
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, alt
        )
        self.log(f"TAKEOFF {alt}m command sent")

    def on_close(self):
        self.running = False
        if self.conn:
            self.conn.close()
        self.root.destroy()


if __name__ == '__main__':
    DroneUI()
