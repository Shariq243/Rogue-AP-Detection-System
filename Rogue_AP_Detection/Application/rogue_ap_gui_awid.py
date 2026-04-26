import os     #for non-root users(os.chmod)
import sys    #for PyInstaller path resolution
import time   #for time.sleep (for delay)
import glob   #for finding files
import subprocess
import threading
import datetime
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.widgets.scrolled import ScrolledText
from tkinter import messagebox

if os.geteuid() != 0:
    print("[!] Critical Error: This tool requires root privileges to manage wireless interfaces.")
    print("    Please run again with: sudo python3 rogue_ap_gui_awid.py")
    sys.exit(1)

def nuke_string(s):
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()
    # Fallback if the string is completely made of special characters or emojis (like "..", "!!!!")
    # This ensures ".." doesn't become "", which breaks matching logic
    if not cleaned and str(s).strip():
        return str(s).strip().lower()
    return cleaned


# Configuration

# Resolve the base directory for data files (works for both script and executable)
if getattr(sys, 'frozen', False):
    _BASE_DIR = Path(sys._MEIPASS)  # PyInstaller temp extraction folder
else:
    _BASE_DIR = Path(__file__).resolve().parent  # Normal script directory

TSHARK_BIN = "tshark"
NMCLI_BIN = "nmcli"
MODEL_PATH = str(_BASE_DIR / "evil_twin_champion_model.pkl")
MODEL_PATH_ACTIVE = str(_BASE_DIR / "dhcp_dns_rf.pkl")  # Phase 3 Brain

DEFAULT_MONITOR = "wlan0"
DEFAULT_DURATION = 15

CAPTURES_DIR = Path("/tmp/FYP_Captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(CAPTURES_DIR, 0o777)
except PermissionError:
    pass  # Directory already exists from a previous root-level run


class RogueAPScanner(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Targeted Evil Twin Detector")
        self.geometry("1200x800")
        self.minsize(900, 600) # Prevents the window from getting too small

        self.is_scanning = False
        self.rf_model = None
        self.scaler = None
        self.latest_df = pd.DataFrame() # To store raw features for the popup
        self.required_features = [
            'frame.time_delta', 'frame.time_delta_displayed', 'frame.time_relative', 
            'radiotap.present.rate', 'radiotap.present.dbm_antsignal', 'radiotap.present.antenna', 
            'radiotap.present.rtap_ns', 'radiotap.present.ext', 'radiotap.datarate', 
            'wlan.fc.type_subtype', 'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.ds', 
            'wlan.qos.tid', 'wlan.qos.priority'
        ]
        self.rf_active_model = None # Ensures Phase 3 doesn't crash
        self.golden_profile = {"ssid": "", "bssid": ""}
        self._hopping = False
        
        # --- Frame Management (Single Page App style) ---
        self.container = tb.Frame(self)
        self.container.pack(fill=BOTH, expand=True)
        
        self.frames = {}
        
        # Initialize the two pages
        self.build_landing_page()
        self.build_dashboard_page()
        
        # Load AI and show landing page
        self.load_model()
        self.show_frame("LandingPage")

    def show_frame(self, page_name):
        """Hides all frames and packs the requested one (Responsive)"""
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[page_name].pack(fill=BOTH, expand=True)

    def load_model(self):
        try:
            bundle = joblib.load(MODEL_PATH)
            self.rf_model = bundle['model'] # Keeping variable name same to prevent downstream breaks
            self.scaler = bundle['scaler']
            self.required_features = bundle.get('features', self.required_features)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load Brain 1: {e}")
            
        try:
            self.rf_active_model = joblib.load(MODEL_PATH_ACTIVE)
        except Exception as e:
            messagebox.showwarning("Model Warning", f"Brain 2 (DHCP/DNS) not loaded: {e}\nPhase 3 will still work but with reduced accuracy.")

   
    # PAGE 1: THE LANDING PAGE
    
    def build_landing_page(self):
        frame = tb.Frame(self.container, padding=40)
        self.frames["LandingPage"] = frame
        
        # Centering wrapper
        center_wrapper = tb.Frame(frame)
        center_wrapper.pack(expand=True)

        tb.Label(center_wrapper, text="🛡️ Evil Twin Detector", font=("Helvetica", 28, "bold")).pack(pady=(0, 10))
        tb.Label(center_wrapper, text="Select your Trusted Network to establish a Golden Profile.", font=("Helvetica", 12), bootstyle="secondary").pack(pady=(0, 20))

        # Wi-Fi Listbox
        self.wifi_listbox = tb.Treeview(center_wrapper, columns=("ssid", "bssid", "signal", "security"), show="headings", height=8, bootstyle="info")
        self.wifi_listbox.heading("ssid", text="Network Name")
        self.wifi_listbox.heading("bssid", text="MAC Address")
        self.wifi_listbox.heading("signal", text="Signal")
        self.wifi_listbox.heading("security", text="Security")
        self.wifi_listbox.column("ssid", width=250)
        self.wifi_listbox.column("bssid", width=180, anchor=CENTER)
        self.wifi_listbox.column("signal", width=80, anchor=CENTER)
        self.wifi_listbox.column("security", width=100, anchor=CENTER)
        self.wifi_listbox.pack(fill=X, pady=10)
        
        self.wifi_listbox.bind("<<TreeviewSelect>>", self.on_wifi_select)

        # Controls
        input_frame = tb.Frame(center_wrapper)
        input_frame.pack(fill=X, pady=10)
        
        tb.Label(input_frame, text="Password:").pack(side=LEFT, padx=5)
        self.ent_pwd = tb.Entry(input_frame, show="*", width=20)
        self.ent_pwd.pack(side=LEFT, padx=5)
        
        self.btn_connect = tb.Button(input_frame, text="Connect & Lock", bootstyle="success", command=self.connect_and_lock)
        self.btn_connect.pack(side=LEFT, padx=10)
        tb.Button(input_frame, text="↻ Refresh List", bootstyle="outline-info", command=self.scan_local_wifi).pack(side=LEFT, padx=5)
        
        # Landing page status label (visible feedback during connection)
        self.lbl_landing_status = tb.Label(center_wrapper, text="", font=("Helvetica", 11, "bold"))
        self.lbl_landing_status.pack(pady=(10, 0))

        # Skip Option
        tb.Separator(center_wrapper, bootstyle="secondary").pack(fill=X, pady=20)
        tb.Button(center_wrapper, text="Skip directly to Dashboard ➔", bootstyle="link-secondary", command=lambda: self.show_frame("DashboardPage")).pack()

        # Auto-scan on startup
        self.after(500, self.scan_local_wifi)

    def scan_local_wifi(self):
        """Uses nmcli to find nearby networks for the landing page"""
        for i in self.wifi_listbox.get_children(): self.wifi_listbox.delete(i)
        
        def run_scan():
            try:
                # Force rescan
                subprocess.run([NMCLI_BIN, "dev", "wifi", "rescan"], check=False)
                
                # Extract SSID, BSSID, SIGNAL, SECURITY
                result = subprocess.run(
                    [NMCLI_BIN, "-t", "-f", "SSID,BSSID,SIGNAL,SECURITY", "dev", "wifi"], 
                    capture_output=True, text=True
                )
                
                seen = set()
                for line in result.stdout.split('\n'):
                    if not line: continue
                    
                    # nmcli escapes MAC colons like 00\:11\:22. We replace them temporarily.
                    clean_line = line.replace('\\:', '-') 
                    parts = clean_line.split(':')
                    
                    if len(parts) >= 4:
                        ssid = parts[0]
                        bssid = parts[1].replace('-', ':') # Put the colons back
                        signal = parts[2]
                        security = parts[3] if len(parts) > 3 else ""
                        
                        # Show lock icon for password-protected networks
                        lock_display = f"🔒 {security}" if security and security != "--" else "Open"
                        
                        if ssid and ssid != "--" and ssid not in seen:
                            self.wifi_listbox.insert("", END, values=(ssid, bssid, signal, lock_display))
                            seen.add(ssid)
                            
            except Exception as e:
                self.log_write(f"[!] nmcli error: {e}")
                print("nmcli error:", e)
                
        threading.Thread(target=run_scan, daemon=True).start()

    def on_wifi_select(self, event):
        self.ent_pwd.delete(0, END)
        self.ent_pwd.focus()

    def connect_and_lock(self):
        selected = self.wifi_listbox.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a network from the list.")
            return
            
        item = self.wifi_listbox.item(selected[0])['values']
        raw_ssid = str(item[0])
        ssid = raw_ssid[:-2] if raw_ssid.endswith(".0") and raw_ssid[:-2].isdigit() else raw_ssid
        bssid = str(item[1])
        pwd = self.ent_pwd.get()
        
        if not pwd:
            messagebox.showwarning("Warning", "Please enter the Wi-Fi password.")
            return

        # Disable button and show status on the LANDING PAGE itself
        self.btn_connect.config(state=DISABLED)
        self.lbl_landing_status.config(text=f"⏳ Connecting to {ssid}...", bootstyle="warning")
        self.log_write(f"[*] Attempting to connect to {ssid}...")

        # Run the blocking nmcli work on a background thread
        threading.Thread(target=self._connect_thread, args=(ssid, bssid, pwd), daemon=True).start()

    def _connect_thread(self, ssid, bssid, pwd):
        """Background thread for nmcli connection — keeps the GUI responsive."""
        try:
            # Delete any old, corrupted NetworkManager profiles for this SSID
            subprocess.run([NMCLI_BIN, "connection", "delete", ssid], capture_output=True)
            time.sleep(1)

            cmd = [NMCLI_BIN, "dev", "wifi", "connect", ssid, "bssid", bssid, "password", pwd]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if "successfully activated" in result.stdout or "successfully" in result.stdout:
                self.golden_profile["ssid"] = str(ssid).lower()
                self.golden_profile["bssid"] = str(bssid).upper()
                
                # Update UI on the main thread via after()
                self.after(0, lambda: self.lbl_trusted_status.config(
                    text=f"🔒 Trusted Anchor Locked: {ssid} ({bssid})", bootstyle="success"))
                self.after(0, lambda: self.lbl_landing_status.config(
                    text=f"✅ Connected to {ssid}!", bootstyle="success"))
                self.after(0, lambda: messagebox.showinfo("Success", f"Connected to {ssid}!\nGolden Profile Locked."))
                self.after(0, lambda: self.show_frame("DashboardPage"))
                self.log_write(f"[+] Golden Profile established: {ssid}")
            else:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                self.after(0, lambda: self.lbl_landing_status.config(
                    text=f"❌ Connection failed.", bootstyle="danger"))
                self.after(0, lambda: messagebox.showerror("Connection Failed", error_msg))
                self.log_write(f"[!] Connection failed: {error_msg}")

        except subprocess.TimeoutExpired:
            self.after(0, lambda: self.lbl_landing_status.config(
                text=f"❌ Connection timed out after 30s.", bootstyle="danger"))
            self.after(0, lambda: messagebox.showerror("Timeout", f"nmcli timed out trying to connect to {ssid}."))
            self.log_write(f"[!] Connection to {ssid} timed out.")
        except Exception as e:
            self.after(0, lambda: self.lbl_landing_status.config(
                text=f"❌ Error: {e}", bootstyle="danger"))
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to run nmcli: {e}"))
            self.log_write(f"[!] nmcli exception: {e}")
        finally:
            self.after(0, lambda: self.btn_connect.config(state=NORMAL))

    
    # PAGE 2: THE DASHBOARD (Responsive)
    
    def build_dashboard_page(self):
        frame = tb.Frame(self.container, padding=10)
        self.frames["DashboardPage"] = frame

        # --- HEADER ---
        header_frame = tb.Frame(frame)
        header_frame.pack(fill=X, pady=(0, 10))
        
        self.lbl_trusted_status = tb.Label(header_frame, text="⚠️ No Trusted Anchor Set (Skpped)", font=("Helvetica", 14, "bold"), bootstyle="warning")
        self.lbl_trusted_status.pack(side=LEFT)
        
        tb.Button(header_frame, text="← Back to Settings", bootstyle="link", command=lambda: self.show_frame("LandingPage")).pack(side=RIGHT)
        tb.Button(header_frame, text="💾 Export Results", bootstyle="success", command=self.export_results).pack(side=RIGHT, padx=10)
        tb.Button(header_frame, text="🗑️ Clear Dashboard", bootstyle="danger-outline", command=self.clear_dashboard).pack(side=RIGHT, padx=10)

        # --- CONTROLS ---
        ctrl_frame = tb.LabelFrame(frame, text=" Hunt Parameters ")
        ctrl_frame.pack(fill=X, pady=5)
        
        inner_ctrl = tb.Frame(ctrl_frame)
        inner_ctrl.pack(fill=X, padx=10, pady=10)

        tb.Label(inner_ctrl, text="Monitor Iface:").pack(side=LEFT, padx=5)
        self.ent_monitor = tb.Entry(inner_ctrl, width=12)
        self.ent_monitor.insert(0, DEFAULT_MONITOR)
        self.ent_monitor.pack(side=LEFT, padx=5)

        tb.Label(inner_ctrl, text="Time (s):").pack(side=LEFT, padx=5)
        self.ent_duration = tb.Entry(inner_ctrl, width=6)
        self.ent_duration.insert(0, str(DEFAULT_DURATION))
        self.ent_duration.pack(side=LEFT, padx=5)

        self.btn_scan = tb.Button(inner_ctrl, text="▶ Hunt for Clones", bootstyle="primary", command=self.start_scan)
        self.btn_scan.pack(side=LEFT, padx=15)

        self.lbl_timer = tb.Label(inner_ctrl, text="Status: Idle", font=("Helvetica", 11, "bold"), bootstyle="info")
        self.lbl_timer.pack(side=LEFT, padx=20)
        
        self.progress = tb.Progressbar(inner_ctrl, mode='indeterminate', bootstyle="warning", length=150)
        # We pack it later when scanning starts

        # --- RESPONSIVE TABLE ---
        # The table frame expands in all directions
        table_frame = tb.Frame(frame)
        table_frame.pack(fill=BOTH, expand=True, pady=10)

        columns = ("ssid", "bssid", "total_pkts", "rogue_pkts", "rogue_pct", "verdict")
        self.tree = tb.Treeview(table_frame, columns=columns, show="headings", bootstyle="info")
        
        self.tree.heading("ssid", text="SSID")
        self.tree.heading("bssid", text="BSSID (MAC)")
        self.tree.heading("total_pkts", text="Packets")
        self.tree.heading("rogue_pkts", text="Rogue Hits")
        self.tree.heading("rogue_pct", text="Rogue %")
        self.tree.heading("verdict", text="Status / Verdict")

        self.tree.column("ssid", anchor=W, width=200)
        self.tree.column("bssid", anchor=CENTER, width=150)
        self.tree.column("total_pkts", anchor=CENTER, width=100)
        self.tree.column("rogue_pkts", anchor=CENTER, width=100)
        self.tree.column("rogue_pct", anchor=CENTER, width=100)
        self.tree.column("verdict", anchor=CENTER, width=300)

        
        self.tree.tag_configure("trusted", background="#343a40", foreground="#20c997")  # Dark Grey, Green Text (Your Network)
        self.tree.tag_configure("clone", background="#721c24", foreground="white")      # Deep Red (The Evil Twin)
        self.tree.tag_configure("normal", background="#1e1e1e", foreground="#adb5bd")   # Faded (Irrelevant neighbors)
        
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self.tree.bind("<Button-3>", self.show_context_menu) # Right-click
        
        # Context Menu for right-click on the dashboard treeview
        self.context_menu = tb.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Set as Trusted Anchor", command=self.set_trusted_anchor_from_dashboard)

        self.tree.pack(fill=BOTH, expand=True, side=LEFT)
        sb = tb.Scrollbar(table_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=RIGHT, fill=Y)

        # --- ACTIVE CHECK ---
        self.btn_active = tb.Button(frame, text="⚡ Target Selected Clone -> Run DHCP Interrogation", bootstyle="warning", command=self.run_active_check)
        self.btn_active.pack(fill=X, pady=5)

        # --- LOGS ---
        self.log_box = ScrolledText(frame, height=8, wrap=WORD, font=("Consolas", 10), bg="#000000", fg="#00ff00")
        self.log_box.pack(fill=X, pady=5)

    def log_write(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert(END, f"[{ts}] {msg}\n")
        self.log_box.see(END)

    def clear_dashboard(self):
        """Empties the dashboard treeview and log box"""
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.log_box.delete("1.0", END)
        self.latest_df = pd.DataFrame()
        self.lbl_timer.config(text="Status: Idle", bootstyle="info")
        
    def export_results(self):
        """Exports the current treeview contents to a CSV file"""
        if not self.tree.get_children():
            messagebox.showinfo("Export", "Dashboard is empty. Nothing to export.")
            return
            
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"RogueAP_Report_{ts}.csv"
        
        path = CAPTURES_DIR / filename
        
        rows = []
        for child in self.tree.get_children():
            rows.append(self.tree.item(child)["values"])
            
        columns = ["SSID", "BSSID", "Total Packets", "Rogue Hits", "Rogue %", "Verdict"]
        df_export = pd.DataFrame(rows, columns=columns)
        
        try:
            df_export.to_csv(path, index=False)
            messagebox.showinfo("Success", f"Results exported successfully to:\n{path}")
            self.log_write(f"[+] Results exported to {path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {e}")
            
    def show_context_menu(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.tk_popup(event.x_root, event.y_root)

    def set_trusted_anchor_from_dashboard(self):
        """Allows setting the golden profile via right-click without password auth"""
        selected = self.tree.selection()
        if not selected: return
        item = self.tree.item(selected[0])['values']
        raw_ssid = str(item[0])
        ssid = raw_ssid[:-2] if raw_ssid.endswith(".0") and raw_ssid[:-2].isdigit() else raw_ssid
        bssid = str(item[1])
        
        self.golden_profile["ssid"] = ssid.lower()
        self.golden_profile["bssid"] = bssid.upper()
        
        self.lbl_trusted_status.config(text=f"🔒 Trusted Anchor Locked: {ssid} ({bssid})", bootstyle="success")
        messagebox.showinfo("Success", f"Trusted Anchor manually updated to {ssid} ({bssid}).\nTracking rules will now apply to this network.")
        self.log_write(f"[+] Trusted Anchor manually updated to {bssid}")

    def on_tree_double_click(self, event):
        """Pops up a window displaying the exact ML feature values for the selected BSSID"""
        selected = self.tree.selection()
        if not selected: return
        
        item = self.tree.item(selected[0])['values']
        bssid = str(item[1]).upper()
        
        if self.latest_df.empty:
            messagebox.showinfo("Wait", "Raw features not yet populated. Please run a hunt first.")
            return
            
        # Get the latest row for this BSSID
        bssid_data = self.latest_df[self.latest_df['wlan.bssid'] == bssid.lower()]
        if bssid_data.empty:
            messagebox.showinfo("No Data", f"No raw ML features found for {bssid} in the latest capture.")
            return
            
        latest_row = bssid_data.iloc[-1]
        
        popup = tb.Toplevel(self)
        popup.title(f"Extracted Features for {bssid}")
        popup.geometry("600x400")
        
        lbl = tb.Label(popup, text=f"Raw Model Inputs for {bssid}", font=("Helvetica", 14, "bold"))
        lbl.pack(pady=10)
        
        cols = ("feature", "value")
        feat_tree = tb.Treeview(popup, columns=cols, show="headings", bootstyle="primary")
        feat_tree.heading("feature", text="Feature Name")
        feat_tree.heading("value", text="Extracted Value")
        feat_tree.column("feature", width=350, anchor=W)
        feat_tree.column("value", width=150, anchor=CENTER)
        
        for feat in self.required_features:
            val = latest_row.get(feat, "N/A")
            # Format nicely if it's a float
            if isinstance(val, (float, np.float32, np.float64)):
                val = f"{val:.6f}"
            feat_tree.insert("", END, values=(feat, val))
            
        feat_tree.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))

  
    # CORE LOGIC: SCAN & PROCESS
    
    
    def start_scan(self):
        iface = self.ent_monitor.get().strip()
        self.log_write(f"[*] Sniffing on {iface}. Internet on wlan1 remains ACTIVE.")
        

        self.btn_scan.config(state=DISABLED)
        self.progress.pack(side=LEFT, padx=10)
        self.progress.start(10) # Start indeterminate spinning
        
        for i in self.tree.get_children(): self.tree.delete(i)
        
        try:
            self.time_left = int(self.ent_duration.get().strip())
        except ValueError:
            self.time_left = DEFAULT_DURATION
            
        self.countdown_active = True
        self.update_timer()
        threading.Thread(target=self.scan_thread, daemon=True).start()

    def update_timer(self):
        if not self.countdown_active: return
        if self.time_left > 0:
            self.lbl_timer.config(text=f"Hunting... {self.time_left}s remaining", bootstyle="warning")
            self.time_left -= 1
            self.after(1000, self.update_timer)
        else:
            self.lbl_timer.config(text="Analyzing Physics Data...", bootstyle="success")

            
    def scan_thread(self):
        iface = self.ent_monitor.get().strip()
        dur = self.ent_duration.get().strip()
        ts = time.strftime("%Y%m%d_%H%M%S")
        
        pcap = CAPTURES_DIR / f"raw_{ts}.pcap"
        csv_file = CAPTURES_DIR / f"raw_{ts}.csv"

        try:
            # Check Directory Integrity
            if not os.path.exists(CAPTURES_DIR):
                os.makedirs(CAPTURES_DIR, exist_ok=True)

            # --- Step 1: Surgical Interface Isolation ---
            self.log_write(f"[*] Isolating {iface} from NetworkManager...")
            subprocess.run([NMCLI_BIN, "device", "disconnect", iface], capture_output=True)
            time.sleep(1)
            subprocess.run([NMCLI_BIN, "device", "set", iface, "managed", "no"], capture_output=True)
            time.sleep(1)

            # Hardware Freeze Prevention (USB Power Cycle)
            try:
                p_bus = subprocess.run(["ethtool", "-i", iface], capture_output=True, text=True)
                bus_info = None
                for line in p_bus.stdout.split('\n'):
                    if line.startswith("bus-info:"):
                        bus_info = line.split()[1].split(':')[0]
                        break
                if bus_info:
                    self.log_write(f"[*] Performing USB unbind/bind on bus {bus_info} to prevent driver freeze...")
                    subprocess.run(f"echo {bus_info} > /sys/bus/usb/drivers/usb/unbind", shell=True)
                    time.sleep(1)
                    subprocess.run(f"echo {bus_info} > /sys/bus/usb/drivers/usb/bind", shell=True)
                    time.sleep(3) # Wait for kernel network layer to re-initialize
            except Exception as e:
                self.log_write(f"[!] Warning: Could not perform USB bind/unbind: {e}")

            # Ensure interface is UP and in monitor mode before capture
            subprocess.run(["rfkill", "unblock", "wifi"], capture_output=True)
            
            p_type = subprocess.run(["iw", "dev", iface, "info"], capture_output=True, text=True)
            if "type monitor" not in p_type.stdout:
                self.log_write(f"[*] Setting {iface} to monitor mode...")
                subprocess.run(["ip", "link", "set", iface, "down"], capture_output=True)
                p_mon = subprocess.run(["iw", "dev", iface, "set", "type", "monitor"], capture_output=True, text=True)
                if p_mon.returncode != 0:
                    self.log_write(f"[!] Failed to set Monitor Mode: {p_mon.stderr.strip()}")
            else:
                self.log_write(f"[*] {iface} is already in monitor mode.")
                
            # Verify monitor mode actually worked
            p_type2 = subprocess.run(["iw", "dev", iface, "info"], capture_output=True, text=True)
            if "type monitor" not in p_type2.stdout:
                self.log_write(f"[!] Critical: {iface} is NOT in monitor mode. Capture will likely fail.")
                
            self.log_write(f"[*] Forcing {iface} UP...")
            subprocess.run(["ip", "link", "set", iface, "up"], capture_output=True)
            subprocess.run(["ifconfig", iface, "up"], capture_output=True)
            
            # Wait for kernel to stabilize
            time.sleep(2)
            
            # Double check if it actually went up
            p_check = subprocess.run(["ip", "link", "show", iface], capture_output=True, text=True)
            if "state UP" not in p_check.stdout and ",UP" not in p_check.stdout:
                 self.log_write(f"[!] Warning: Kernel reports {iface} is still DOWN. tshark may fail.")

            self.log_write(f"[*] Sniffing airwaves on {iface} for {dur}s...")

            # --- Step 2: Capture with tshark ---
            # tshark does not channel-hop by default like airodump-ng. Start a hopper thread.
            self._hopping = True
            def hopper():
                chans = [1, 6, 11, 2, 7, 3, 8, 4, 9, 5, 10]
                idx = 0
                while self._hopping:
                    subprocess.run(["iw", "dev", iface, "set", "channel", str(chans[idx])], capture_output=True)
                    idx = (idx + 1) % len(chans)
                    time.sleep(0.5)

            hop_thread = threading.Thread(target=hopper, daemon=True)
            hop_thread.start()

            p = subprocess.run([TSHARK_BIN, "-i", iface, "-a", f"duration:{dur}",
                                "-w", str(pcap)],
                               capture_output=True, text=True)

            self._hopping = False
            hop_thread.join(timeout=2)

            if p.returncode != 0:
                self.log_write(f"[!] tshark failed: {p.stderr.strip()}")
                return
                
            # Parse stderr to show how many raw packets were captured
            dump_stats = "Unknown"
            if p.stderr:
                for line in p.stderr.split("\n"):
                    if "packet" in line.lower() and "captured" in line.lower():
                        dump_stats = line.strip()
            self.log_write(f"[*] Capture complete. ({dump_stats}) Processing packets...")

             # --- Step 3: Extract beacon-frame fields into CSV ---
            cmd_ext = [
                TSHARK_BIN, "-r", str(pcap), "-Y", "wlan.fc.type_subtype==8",
                "-T", "fields", "-E", "header=y", "-E", "separator=,", "-E", "quote=d"
            ]
            for feat in self.required_features:
                cmd_ext.extend(["-e", feat])
            cmd_ext.extend(["-e", "wlan.bssid", "-e", "wlan.ssid"])
            with open(csv_file, "w") as f:
                subprocess.run(cmd_ext, stdout=f, check=True)
                
            self.process_data(csv_file)
            
        except Exception as e:
            self.log_write(f"[!] Scan Error: {e}")
        finally:
            self.countdown_active = False
            self.btn_scan.config(state=NORMAL)
            self.progress.stop()
            self.progress.pack_forget()
            self.lbl_timer.config(text="Status: Idle", bootstyle="info")

    def decode_hex_ssid(self, val):
        """Convert hex-encoded SSID from tshark back to a readable string."""
        try:
            s = str(val).strip()
            if s and all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0 and len(s) >= 2:
                return bytes.fromhex(s).decode('utf-8', errors='replace')
            return s
        except Exception:
            return str(val)

    def process_data(self, path):
        try:
            df = pd.read_csv(path).dropna(subset=['wlan.bssid'])
            if df.empty:
                self.log_write("[!] No packets captured.")
                return

            if 'wlan_radio.channel' in df.columns:
                df.rename(columns={'wlan_radio.channel': 'wlan_mgt.ds.current_channel'}, inplace=True)
            if 'wlan.ssid' in df.columns:
                df['wlan.ssid'] = df['wlan.ssid'].apply(self.decode_hex_ssid)

            def clean_hex(val):
                if isinstance(val, str):
                    clean_str = val.strip().lower()
                    if clean_str == 'true':
                        return 1.0
                    if clean_str == 'false':
                        return 0.0
                    if val.startswith("0x"):
                        try:
                            return int(val, 16)
                        except ValueError:
                            return np.nan
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return np.nan

            for feat in self.required_features:
                if feat not in df.columns:
                    df[feat] = 0
                else:
                    df[feat] = df[feat].apply(clean_hex).fillna(0)
                    
            # Save the processed raw dataframe for the View Features popup
            self.latest_df = df.copy()
            
            if self.rf_model and self.scaler:
                X_live = df[self.required_features].to_numpy().astype(np.float32)
                X_scaled = self.scaler.transform(X_live)
                df['is_rogue'] = self.rf_model.predict(X_scaled)
            else:
                df['is_rogue'] = 0
            
            gp = df.groupby('wlan.bssid').agg(
                total=('is_rogue', 'count'), rogues=('is_rogue', 'sum'),
                ssid_list=('wlan.ssid', lambda x: [str(i).strip() for i in x.dropna() if str(i).strip() and str(i).strip() != "nan"])
            ).reset_index()

            for _, r in gp.iterrows():
                mac = str(r['wlan.bssid']).upper() 
                ssid = r['ssid_list'][0] if r['ssid_list'] else "<Hidden>"
                pct = (r['rogues'] / r['total']) * 100
                
                # Default for neighbors (Irrelevant)
                tag = "normal"
                verdict = "Neighbor AP (Ignored)"
                
                # THE TARGETED LOGIC
                gold_ssid = self.golden_profile.get("ssid", "")
                gold_mac = self.golden_profile.get("bssid", "")

                if gold_ssid and gold_mac:
                    # Strip absolutely everything except letters and numbers
                    clean_scanned = nuke_string(ssid)
                    clean_golden = nuke_string(gold_ssid)

                    # Tshark often hex encodes short strings. Check if decoding matches too.
                    try:
                        clean_decoded = nuke_string(bytes.fromhex(clean_scanned).decode('utf-8', errors='ignore'))
                    except Exception:
                        clean_decoded = clean_scanned

                    # First, check if the network is claiming to be our trusted SSID
                    if clean_scanned == clean_golden or (clean_golden != "" and clean_decoded == clean_golden):
                        
                        # 1. Trust the AI first: Is the radio physics/behavior malicious?
                        if pct > 40: 
                            tag = "clone"
                            # Did they also perfectly spoof the MAC address?
                            if mac == gold_mac:
                                verdict = f"🚨 ADVANCED EVIL TWIN (MAC Spoofed! AI Score: {pct:.0f}%)"
                            else:
                                verdict = f"🚨 CONFIRMED EVIL TWIN (AI Score: {pct:.0f}%)"
                                
                        # 2. AI thinks the behavior is normal, so now check the MAC identity
                        else:
                            if mac == gold_mac:
                                tag = "trusted"
                                verdict = "YOUR TRUSTED NETWORK"
                            else:
                                tag = "clone"
                                verdict = "🚨 SUSPICIOUS CLONE (MAC Mismatch)"
                                
                self.tree.insert("", END, values=(ssid, mac, r['total'], r['rogues'], f"{pct:.1f}%", verdict), tags=(tag,))
            
            self.log_write("[+] Hunt complete. Dashboard updated.")
        except Exception as e:
            self.log_write(f"[!] Process error: {e}")

    def run_active_check(self):
        sel = self.tree.selection()
        if not sel: 
            messagebox.showwarning("Warning", "Select a Clone to interrogate.")
            return
            
        item = self.tree.item(sel[0])['values']
        raw_ssid = str(item[0])
        target_ssid = raw_ssid[:-2] if raw_ssid.endswith(".0") and raw_ssid[:-2].isdigit() else raw_ssid
        target_mac = str(item[1])

        # ALWAYS ask for the rogue AP's password (it's a different network than the trusted one)
        from tkinter import simpledialog
        rogue_pwd = simpledialog.askstring(
            "Rogue AP Password", 
            f"Enter Wi-Fi password for the CLONE '{target_ssid}':\n(Leave BLANK for open networks, then click OK)", 
            parent=self, show='*'
        )
        if rogue_pwd is None:  # User ki marzi h is ny ok y cancel kr diya
            self.log_write("[!] Phase 3 aborted by user.")
            return
        # rogue_pwd will be "" for open networks, or the actual password

        self.log_write("\n" + "="*50)
        self.log_write(f"[*] INITIATING PHASE 3: ACTIVE INTERROGATION")
        self.log_write(f"[*] Target: {target_ssid} ({target_mac})")
        self.log_write("[!] INITIATING SECURITY PROTOCOL: AIR-GAPPING SYSTEM...")
        
        self.btn_active.config(state=DISABLED)
        self.progress.pack(side=LEFT, padx=10)
        self.progress.start(5)
        
        # Pass the rogue password directly to the thread
        threading.Thread(target=self.active_interrogation_thread, args=(target_ssid, target_mac, rogue_pwd), daemon=True).start()

    def active_interrogation_thread(self, target_ssid, target_mac, rogue_pwd):
        managed_iface = "wlan1"
        
        try:
            # 1. THE SECURITY AIR-GAP
            self.log_write("\n" + "="*50)
            self.log_write("[!] SECURITY PROTOCOL: Severing all system connections...")
            subprocess.run([NMCLI_BIN, "radio", "wifi", "off"], check=False)
            time.sleep(1)
            subprocess.run([NMCLI_BIN, "radio", "wifi", "on"], check=False)
            time.sleep(3)  # Give radio time to come back up

            self.log_write("[*] Forcing NetworkManager to rescan airwaves...")
            
            # Wait until the target BSSID appears in NM's scan list
            bssid_found = False
            for _ in range(15):
                subprocess.run([NMCLI_BIN, "dev", "wifi", "rescan", "ifname", managed_iface], capture_output=True)
                time.sleep(2)
                scan_out = subprocess.run([NMCLI_BIN, "dev", "wifi", "list", "ifname", managed_iface], capture_output=True, text=True)
                if target_mac.upper() in scan_out.stdout.upper():
                    bssid_found = True
                    break
                    
            if not bssid_found:
                self.log_write(f"[!] Critical: Target BSSID {target_mac} not found in NetworkManager scan dump. Aborting connection.")
                return
                
            self.log_write(f"[+] Target {target_mac} is visible to NetworkManager.")

            # Clean up any leftover probe profile from a previous run
            subprocess.run([NMCLI_BIN, "connection", "delete", "ET-Probe"],
                           capture_output=True, check=False)
            
            # 2. DIRECT CONNECTION (bypasses keyring issues under sudo)
            self.log_write(f"[*] Connecting to Target by MAC: {target_mac}...")
            pwd = rogue_pwd.strip() if rogue_pwd else ""
            
            # Build the connection command: with or without password
            if pwd:
                # Password-protected network
                connect_cmd = [NMCLI_BIN, "dev", "wifi", "connect", target_ssid, 
                               "bssid", target_mac, "password", pwd, "ifname", managed_iface]
            else:
                # Open network (no password)
                connect_cmd = [NMCLI_BIN, "dev", "wifi", "connect", target_ssid, 
                               "bssid", target_mac, "ifname", managed_iface]
            
            result_up = subprocess.run(connect_cmd, capture_output=True, text=True, timeout=45)
            
            if "successfully" in result_up.stdout.lower():
                self.log_write("[+] System Air-Gapped and Connected to Rogue AP.")
                time.sleep(2)  # Let DHCP finish
                
                # 3. LIVE DHCP/DNS EXTRACTION
                self.log_write("[*] Extracting DHCP & DNS Telemetry...")
                nmcli_show = subprocess.run([NMCLI_BIN, "dev", "show", managed_iface], capture_output=True, text=True)
                
                assigned_ip = "Unknown"
                current_dns = "Unknown"
                current_gw  = "Unknown"
                
                for line in nmcli_show.stdout.split('\n'):
                    if "IP4.ADDRESS[1]:" in line:
                        assigned_ip = line.split()[1].split('/')[0]
                    elif "IP4.DNS[1]:" in line:
                        current_dns = line.split()[1]
                    elif "IP4.GATEWAY:" in line:
                        current_gw = line.split()[1]
                
                self.log_write(f"[>] Assigned IP: {assigned_ip}")
                self.log_write(f"[>] DNS Server: {current_dns}")
                self.log_write(f"[>] Gateway:    {current_gw}")
                
                # --- REAL Brain 2 Feature Extraction via tshark ---
                self.log_write("[*] Capturing live DHCP/DNS traffic (8s)...")
                ts_cap = CAPTURES_DIR / f"phase3_{int(time.time())}.pcap"
                subprocess.run([TSHARK_BIN, "-i", managed_iface, "-a", "duration:8",
                                "-w", str(ts_cap)], capture_output=True)
                
                # Count DHCP Offers
                dhcp_result = subprocess.run(
                    [TSHARK_BIN, "-r", str(ts_cap), "-Y", "dhcp.option.dhcp == 2", "-T", "fields", "-e", "frame.number"],
                    capture_output=True, text=True)
                dhcp_offer_count = len([l for l in dhcp_result.stdout.strip().split('\n') if l.strip()])
                
                # Count DNS Queries
                dns_result = subprocess.run(
                    [TSHARK_BIN, "-r", str(ts_cap), "-Y", "dns", "-T", "fields", "-e", "frame.number"],
                    capture_output=True, text=True)
                dns_queries_total = len([l for l in dns_result.stdout.strip().split('\n') if l.strip()])
                
                self.log_write(f"[>] DHCP Offers captured: {dhcp_offer_count}")
                self.log_write(f"[>] DNS Queries captured: {dns_queries_total}")
                
                # Feed REAL features to Brain 2
                live_dhcp_features = pd.DataFrame([{
                    'dhcp_offer_count': dhcp_offer_count,
                    'dns_queries_total': dns_queries_total
                }])
                
                # BRAIN 2 PREDICTION
                if self.rf_active_model:
                    brain2_prob = self.rf_active_model.predict_proba(live_dhcp_features)[0][1] * 100
                else:
                    brain2_prob = 50.0  # Neutral if model is missing (don't auto-alarm)
                    
                self.log_write(f"[>] Brain 2 Active Threat Score: {brain2_prob:.1f}%")
                
                # --- IP/DNS HEURISTIC CHECKS ---
                ip_dns_suspicious = False
                ip_dns_reasons = []
                
                # Check 1: Suspicious IP ranges (airgeddon defaults to 10.0.0.x)
                if assigned_ip.startswith("10.0.0."):
                    ip_dns_suspicious = True
                    ip_dns_reasons.append("Suspicious IP range (10.0.0.x = common Evil Twin tool)")
                    
                # Check 2: DNS pointing to the gateway (attacker IS the DNS server)
                if current_dns != "Unknown" and current_gw != "Unknown" and current_dns == current_gw:
                    ip_dns_suspicious = True
                    ip_dns_reasons.append("DNS == Gateway (attacker acting as DNS resolver)")
                    
                # Check 3: Non-standard private subnets (unusual for home routers)
                if assigned_ip != "Unknown":
                    parts = assigned_ip.split('.')
                    if len(parts) == 4:
                        # 169.254.x.x = APIPA (no real DHCP server responded)
                        if assigned_ip.startswith("169.254."):
                            ip_dns_suspicious = True
                            ip_dns_reasons.append("APIPA address (no real DHCP server)")
                        # Unusual Class A private (172.16-31.x.x is normal, others aren't)
                        elif parts[0] == "172" and not (16 <= int(parts[1]) <= 31):
                            ip_dns_suspicious = True
                            ip_dns_reasons.append(f"Non-standard 172.x range ({assigned_ip})")
                
                if ip_dns_suspicious:
                    for reason in ip_dns_reasons:
                        self.log_write(f"[!!!] IP/DNS FLAG: {reason}")
                else:
                    self.log_write("[+] IP/DNS Heuristics: No suspicious indicators.")
                
                # --- THE MERGE ---
                selected = self.tree.selection()[0]
                
                old_verdict = self.tree.item(selected)['values'][5]
                is_mac_spoof = "MAC Mismatch" in old_verdict
                
                brain1_str = self.tree.item(selected)['values'][4]
                brain1_prob = float(brain1_str.replace('%', ''))
                
                gold_ssid = self.golden_profile.get("ssid", "")
                
                # Strip absolutely everything except letters and numbers
                clean_scanned_active = nuke_string(target_ssid)
                clean_golden_active = nuke_string(gold_ssid)
                
                name_match_score = 100.0 if clean_scanned_active == clean_golden_active else 0.0
                
                final_merged_score = (name_match_score * 0.1) + (brain1_prob * 0.3) + (brain2_prob * 0.6)
                self.log_write(f"[!!!] FINAL MERGED THREAT PROBABILITY: {final_merged_score:.1f}%")
                
                # Update UI Dashboard with IP, DNS, and Verdict
                if final_merged_score >= 70 or is_mac_spoof:
                    if is_mac_spoof:
                        new_verdict = f"🚨 MAC SPOOF CONFIRMED (IP: {assigned_ip} | DNS: {current_dns})"
                    else:
                        new_verdict = f"🚨 EVIL TWIN (IP: {assigned_ip} | DNS: {current_dns})"
                        
                    self.tree.item(selected, values=(target_ssid, target_mac, "N/A", "N/A", f"{final_merged_score:.1f}%", new_verdict), tags=("clone",))
                    messagebox.showerror("CRITICAL ALERT", f"Phase 3 Complete.\nEvil Twin confirmed!\nIP: {assigned_ip}\nDNS: {current_dns}")
                else:
                    new_verdict = f"✅ Safe (IP: {assigned_ip} | DNS: {current_dns})"
                    self.tree.item(selected, values=(target_ssid, target_mac, "N/A", "N/A", f"{final_merged_score:.1f}%", new_verdict), tags=("trusted",))
                    
                # 4. INSTANT DISCONNECT
                time.sleep(2)
                self.log_write("[*] Interrogation complete. Terminating rogue connection...")
                subprocess.run([NMCLI_BIN, "device", "disconnect", managed_iface], check=False)
                
            else:
                self.log_write(f"[!] Target refused connection: {result_up.stderr.strip()}")
                
        except Exception as e:
            self.log_write(f"[!] Phase 3 Error: {e}")
        finally:
            self.btn_active.config(state=NORMAL)
            self.progress.stop()
            self.progress.pack_forget()
            
            # Always clean up the temporary connection profile
            subprocess.run([NMCLI_BIN, "connection", "delete", target_ssid],
                           capture_output=True, check=False)
            
        self.log_write("="*50 + "\n")

if __name__ == "__main__":
    app = RogueAPScanner()
    app.place_window_center()
    app.mainloop()