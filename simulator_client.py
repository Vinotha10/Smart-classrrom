# simulator_client.py
import time, requests, pandas as pd, argparse
from datetime import datetime
import math

SERVER = "http://127.0.0.1:5000"
DATA_FN = "data/sim_data.csv"

def run(realtime_scale=60.0):
    # realtime_scale: seconds per simulated hour (i.e. 60 => 1 minute per simulated hour)
    df = pd.read_csv(DATA_FN)
    # pick classrooms list
    classes = df['classroom'].unique().tolist()
    print("Classrooms:", classes)
    # battery states per classroom
    battery = {c: 0.5 for c in classes}  # SOC fractional
    # iterate rows in time order
    df = df.sort_values('timestamp')
    for idx, row in df.iterrows():
        payload = {
            "timestamp": row['timestamp'],
            "classroom": row['classroom'],
            "is_holiday": int(row['is_holiday']),
            "scheduled": int(row['scheduled']),
            "occupancy": int(row['occupancy']),
            "motion": int(row['motion']),
            "temp": float(row['temp']),
            "co2": float(row['co2']),
            "solar_kw": float(row['solar_kw']),
        }
        # add battery info
        payload['battery_soc'] = battery[row['classroom']]
        try:
            r = requests.post(SERVER + "/update", json=payload, timeout=5)
            resp = r.json()
            # print a short status
            print(f"[{row['timestamp']} | {row['classroom']}] occ={row['occupancy']}, pred={resp.get('predicted_occupancy')}, devices={resp['control']}, energy={resp['energy']}")
            # update battery: charge if solar > usage, else discharge
            produced = payload['solar_kw']
            used = resp['energy']['total_kwh']
            net = produced - used
            # battery capacity normalized 1.0 => store 5 kWh; here net per hour relative
            # simple SOC update
            battery[row['classroom']] = min(max(battery[row['classroom']] + net*0.02, 0.0), 1.0)
        except Exception as e:
            print("Error posting:", e)
        # sleep: accelerate time: realtime_scale seconds per simulated hour
        time.sleep(max(0.05, realtime_scale/3600.0))  # allow small delay
    print("Simulation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=60.0, help="seconds per simulated hour")
    args = parser.parse_args()
    run(realtime_scale=args.scale)
