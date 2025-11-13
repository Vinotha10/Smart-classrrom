# data_generator.py
import os, json, math, random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_timetable(num_classrooms=4):
    # For each classroom create a weekly schedule (0/1 per hour)
    # simple approach: each day has random 4-6 classes placed across 8am-6pm
    timetable = {}
    for c in range(num_classrooms):
        schedule = {}
        for dow in range(7):  # 0=Mon ... 6=Sun
            slots = [0]*24
            if dow < 5:
                # weekdays
                num_classes = random.randint(4,6)
                hours = random.sample(range(8,18), num_classes)
                for h in hours:
                    slots[h] = 1
            else:
                # weekends fewer classes
                if random.random() < 0.2:
                    slots[random.randint(9,15)] = 1
            schedule[dow] = slots
        timetable[f"class_{c}"] = schedule
    return timetable

def simulate(start_date="2025-10-01", days=14, num_classrooms=4, capacity=30):
    start = datetime.fromisoformat(start_date)
    rows = []
    tt = generate_timetable(num_classrooms)
    for c in range(num_classrooms):
        class_id = f"class_{c}"
        # special events: add one seminar in the period randomly
        seminar_day = random.randint(0, days-1) if random.random() < 0.5 else None
        for d in range(days):
            date = start + timedelta(days=d)
            dow = date.weekday()
            for hour in range(24):
                ts = datetime(date.year, date.month, date.day, hour)
                # holidays: randomly mark a day as holiday
                is_holiday = (random.random() < 0.03)  # ~3% days are holidays
                # base occupancy from timetable
                scheduled = tt[class_id][dow][hour] if not is_holiday else 0
                # special event check
                if seminar_day is not None and d == seminar_day:
                    # if seminar hour, bump occupancy randomly at midday
                    scheduled = 1 if hour in (10,11,14) and random.random() < 0.6 else scheduled

                # occupancy count: if scheduled => mean 0.7*capacity; if unscheduled => small random
                if scheduled:
                    mean = 0.6*capacity + random.uniform(-3, 3)
                    occ = max(0, int(np.random.poisson(max(1, mean))))
                else:
                    # spontaneous occupancy (cleanup staff etc.)
                    occ = int(np.random.poisson(0.2))
                # motion sensor: true if occ > 0 or small prob
                motion = 1 if occ > 0 or random.random() < 0.02 else 0
                # temp: base 25 deg plus 0.06 per person
                temp = 25 + 0.06*occ + random.uniform(-0.5,0.5)
                # co2: baseline 410 ppm + 8 per person + noise
                co2 = 410 + 8*occ + random.uniform(-10, 10)
                # solar generation (kW) approximate sinusoidal over day
                # assume per-site solar capacity 3 kW
                solar_capacity = 3.0
                # solar irradiance factor: 0 at night, peak midday
                irradiance = max(0, math.sin((hour-6)/12*math.pi))  # rough bell curve
                # cloudiness random factor
                cloud = random.uniform(0.6, 1.0)
                solar_kw = solar_capacity * irradiance * cloud
                # battery SOC: simple simulated, we compute later in streaming; here set placeholder
                battery_soc = None
                rows.append({
                    "timestamp": ts.isoformat(),
                    "classroom": class_id,
                    "is_holiday": int(is_holiday),
                    "scheduled": int(scheduled),
                    "occupancy": int(occ),
                    "motion": int(motion),
                    "temp": round(temp,2),
                    "co2": round(co2,1),
                    "solar_kw": round(solar_kw,3)
                })
    df = pd.DataFrame(rows)
    fn = os.path.join(OUT_DIR, "sim_data.csv")
    df.to_csv(fn, index=False)
    print(f"Saved {fn} with {len(df)} rows.")
    return fn

if __name__ == "__main__":
    simulate(start_date="2025-10-01", days=21, num_classrooms=4)
