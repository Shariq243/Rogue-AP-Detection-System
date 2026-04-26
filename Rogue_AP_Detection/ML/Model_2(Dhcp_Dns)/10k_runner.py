#!/usr/bin/env python3
import os
import time
import random
import math
import pandas as pd
import scapy.all as scapy
import traceback
from mininet.node import OVSController
from mn_wifi.net import Mininet_wifi
from mn_wifi.link import wmediumd

from pathlib import Path

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent

REAL_OUIS = ["00:14:22", "C0:C9:E3", "A0:04:60", "08:86:3B", "F8:32:E4", "00:25:00"]
OUTPUT_CSV = str(BASE_DIR / "data" / "Final_10k_Dataset.csv")


# PRE-CONFIGURED FOR 1000 BATCHES
TOTAL_BATCHES = 1000  

def generate_realistic_mac():
    oui = random.choice(REAL_OUIS)
    nic = ":".join([f"{random.randint(0, 255):02X}" for _ in range(3)])
    return f"{oui}:{nic}"

# --- 1. MININET SIMULATION ---
def run_simulation(batch_num):
    print(f"\n==========================================", flush=True)
    print(f"--- Starting Batch {batch_num}/{TOTAL_BATCHES} (5 Real / 5 Rogue) ---", flush=True)
    os.system('mkdir -p /tmp/Data && sudo chmod 777 /tmp/Data')
    os.system('killall dnsmasq tcpdump 2>/dev/null')
    
    net = Mininet_wifi(controller=OVSController, link=wmediumd)
    c0 = net.addController('c0')
    
    target_ssids = ["Campus_WiFi", "Student_Guest", "Library_5G", "Cafe_Free_Web", "Admin_Net"]
    rogue_macs = [] 
    rogue_ssids = []

    # 5 Real APs (FIXED)
    for i in range(5):
        mac = generate_realistic_mac()
        ssid = random.choice(target_ssids)
        net.addAccessPoint(f'n{i}', ssid=ssid, mode='g', channel='1', mac=mac, position=f'{random.randint(10,40)},{random.randint(10,40)},0')
        net.addStation(f'sta_n{i}', position=f'{random.randint(10,40)},{random.randint(10,40)},0')
        
    # 5 Rogue APs (FIXED)
    for i in range(5):
        mac = generate_realistic_mac()
        rogue_macs.append(mac)
        ssid = random.choice(target_ssids) 
        rogue_ssids.append(ssid)
        channel = random.choice(['1', '6', '11']) 
        net.addAccessPoint(f'r{i}', ssid=ssid, mode='g', channel=channel, mac=mac, position=f'{random.randint(10,40)},{random.randint(10,40)},0')
        net.addStation(f'sta_r{i}', position=f'{random.randint(10,40)},{random.randint(10,40)},0')

    net.configureWifiNodes()
    net.build()
    net.start()

    pcap_file = f"/tmp/Data/batch_{batch_num}.pcap"
    os.system('ifconfig hwsim0 up')
    os.system(f'tcpdump -i hwsim0 -U -n -w {pcap_file} 2>/dev/null &')
    
    # Trigger Rogue DHCP & Connect Stations (FIXED LOOP TO 5)
    for i in range(5):
        ap = net.get(f'r{i}')
        ap.cmd(f'ifconfig r{i}-wlan0 10.0.{i}.1 netmask 255.255.255.0 up')
        ap.cmd(f'dnsmasq --interface=r{i}-wlan0 --dhcp-range=10.0.{i}.10,10.0.{i}.50,12h -R --address=/#/10.0.{i}.1 2>/dev/null &')
        
        sta = net.get(f'sta_r{i}')
        sta.cmd(f'iw dev sta_r{i}-wlan0 connect {rogue_ssids[i]}')

    print("  > Waiting 5 seconds for stations to associate...", flush=True)
    time.sleep(5)

    # (FIXED LOOP TO 5)
    for i in range(5):
        sta = net.get(f'sta_r{i}')
        sta.cmd(f'udhcpc -i sta_r{i}-wlan0 -n -t 3 -q 2>/dev/null &')
        sta.cmd(f'nslookup google.com 10.0.{i}.1 2>/dev/null &')

    # Simulate Movement for 20 seconds (FIXED LOOP TO 5)
    for _ in range(20):
        for i in range(5):
            net.get(f'sta_n{i}').setPosition(f'{random.randint(5,45)},{random.randint(5,45)},0')
            net.get(f'sta_r{i}').setPosition(f'{random.randint(5,45)},{random.randint(5,45)},0')
        time.sleep(1)

    os.system('pkill -SIGINT tcpdump')
    time.sleep(2) 
    os.system('killall dnsmasq 2>/dev/null')
    net.stop()
    os.system('sudo mn -c > /dev/null 2>&1')
    return pcap_file, rogue_macs

# --- 2. EXTRACTION MATH ---
def entropy_of_string(s: str):
    if not s: return 0.0
    from collections import Counter
    cnt = Counter(s)
    probs = [v / len(s) for v in cnt.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def extract_features(pcap_path, rogue_macs, batch_num):
    print(f"--- Extracting Batch {batch_num} Features ---", flush=True)
    try: pkts = scapy.rdpcap(pcap_path)
    except Exception as e: 
        print(f"Error reading PCAP: {e}", flush=True)
        return

    rows = []
    
    for pkt in pkts:
        if not pkt.haslayer(scapy.Dot11): continue
        if getattr(pkt, "type", None) == 0 and getattr(pkt, "subtype", None) == 8:
            bssid = str(pkt.addr3).upper() if pkt.addr3 else None
            if not bssid: continue
            
            ts = float(pkt.time)
            ssid, ch = "", 1
            if pkt.haslayer(scapy.Dot11Elt):
                for elt in pkt.iterpayloads():
                    idx = getattr(elt, "ID", None)
                    if idx == 0:
                        try: ssid = elt.info.decode(errors="ignore")
                        except: pass
                    elif idx == 3:
                        try: ch = int.from_bytes(elt.info, "little")
                        except: pass

            rssi = -100
            if pkt.haslayer(scapy.RadioTap):
                try: val = pkt[scapy.RadioTap].fields.get("dBm_AntSignal")
                except: val = None
                if val is not None: rssi = int(val)
                
            rows.append({"ts": ts, "bssid": bssid, "ssid": ssid, "rssi": rssi, "channel": ch})

    dfpk = pd.DataFrame(rows)
    if dfpk.empty: return

    final_rows = []
    for bssid, g in dfpk.groupby("bssid"):
        pkt_count = len(g)
        if pkt_count == 0: continue
        label = 1 if bssid in rogue_macs else 0
        
        scenario = 0 
        if label == 1: scenario = random.choice([1, 2, 3])
            
        rssi_mean, rssi_std = 0, 0
        sequence_number_jumps = random.randint(0, 2)
        beacon_interval_mean = round(random.uniform(102.0, 102.8), 2)
        beacon_interval_jitter = round(random.uniform(0.1, 0.9), 4)
        encryption_type = 1 
        
        if scenario == 1: 
            base_rssi = random.randint(-45, -25)
            encryption_type = random.choice([0, 1])
        elif scenario == 2: 
            base_rssi = random.randint(-70, -50)
            sequence_number_jumps = random.randint(15, 60) 
            beacon_interval_jitter = round(random.uniform(4.0, 12.0), 4) 
        elif scenario == 3: 
            base_rssi = random.randint(-75, -55)
            encryption_type = 1 
        else: 
            base_rssi = random.randint(-75, -50)
            encryption_type = random.choices([0, 1], weights=[0.1, 0.9])[0]

        rssi_vals = [base_rssi + random.randint(-3, 3) for _ in range(pkt_count)]
        rssi_mean = sum(rssi_vals) / len(rssi_vals)
        rssi_std = float(pd.Series(rssi_vals).std()) if len(rssi_vals) > 1 else 0.0

        signal_stability = abs(rssi_std) / max(abs(rssi_mean), 1) if rssi_mean != 0 else 0
        distance = round(10 ** ((-40 - rssi_mean) / (10 * 2.7)), 2) if rssi_mean != 0 else 0

        times = g["ts"].dropna().astype(float).sort_values().tolist()
        duration = times[-1] - times[0] if len(times) >= 2 else 0.0
        b_per_min = (pkt_count / duration * 60.0) if duration > 0 else float(pkt_count)
        
        intervals = [t2 - t1 for t1, t2 in zip(times, times[1:])] if len(times) >= 2 else []
        frame_interval_variance = float(pd.Series(intervals).var()) if len(intervals) > 1 else 0.0
        
        ssid_str = str(g["ssid"].iloc[0])

        retry_frame_count = random.randint(0, max(1, int(pkt_count * 0.05)))
        ht_capabilities = random.choices([0, 1], weights=[0.1, 0.9])[0]
        vht_capabilities = random.choices([0, 1], weights=[0.4, 0.6])[0]
        qos_wmm = random.choices([0, 1], weights=[0.1, 0.9])[0]

        if label == 1: 
            dhcp_offer_count = random.randint(2, 6) 
            dns_queries_total = random.randint(15, 35) 
        else: 
            dhcp_offer_count = random.randint(0, 1) 
            dns_queries_total = random.randint(1, 8)

        final_rows.append({
            "bssid": bssid, "ssid": ssid_str, "label": label, 
            "channel": int(g["channel"].iloc[0]), "rssi_mean": round(rssi_mean, 2),
            "rssi_std": round(rssi_std, 2), "rssi_min": min(rssi_vals),
            "rssi_max": max(rssi_vals), "signal_stability": round(signal_stability, 4),
            "beacon_interval_mean": beacon_interval_mean,
            "beacon_interval_jitter": beacon_interval_jitter, 
            "beacon_count_per_min": round(b_per_min, 2), "frame_interval_variance": round(frame_interval_variance, 4),
            "sequence_number_jumps": sequence_number_jumps, "retry_frame_count": retry_frame_count,
            "encryption_type": encryption_type, "ht_capabilities": ht_capabilities,
            "vht_capabilities": vht_capabilities, "qos_wmm": qos_wmm,
            "ssid_length_entropy": round(entropy_of_string(ssid_str), 4),
            "seconds_visible": round(duration, 2), "distance_estimate": distance,
            "dhcp_offer_count": dhcp_offer_count, "dns_queries_total": dns_queries_total 
        })

    df_final = pd.DataFrame(final_rows)
    dup = df_final.groupby("ssid")["bssid"].nunique().reset_index().rename(columns={"bssid": "duplicate_ssid_count"})
    df_final = df_final.merge(dup, on="ssid", how="left")

    df_final.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
    os.system(f'chmod 777 {OUTPUT_CSV}')

if __name__ == '__main__':
    # IMPORTANT: Commented out so it DOES NOT delete your 6500 rows!
    # if os.path.exists(OUTPUT_CSV):
    #     os.remove(OUTPUT_CSV)
        
    print(f"Starting AP generation...", flush=True)
    
    # RESUMING FROM BATCH 652
    for b in range(652, TOTAL_BATCHES + 1):
        pcap_path = f"/tmp/Data/batch_{b}.pcap"
        try:
            pcap_path, rogues = run_simulation(b)
            extract_features(pcap_path, rogues, b)
        except Exception as e:
            print(f"\n[!] CRITICAL ERROR in Batch {b}: {e}", flush=True)
            traceback.print_exc()
            print("[*] Cleaning up and moving to next batch to prevent total crash...", flush=True)
            os.system('sudo mn -c > /dev/null 2>&1')
            time.sleep(3)
        finally:
            # This is the part Python was complaining about
            if os.path.exists(pcap_path):
                os.remove(pcap_path)
        
    print(f"\n\n!!! COMPLETE !!! Dataset successfully built at: {OUTPUT_CSV}", flush=True)
