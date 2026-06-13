"""
generate_training_data.py
Generates 5 years of statistically realistic Indian Railways delay data.
Mirrors real-world patterns: seasonal variation, cascade propagation,
rush-hour congestion, monsoon impact, track maintenance cycles.

Run: python scripts/generate_training_data.py
Output: data/historical/delays.csv, data/historical/cascade_pairs.csv
"""

import json
import csv
import random
import math
import os
from datetime import datetime, timedelta

random.seed(42)

# ── Load network ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, "data", "network", "stations.json")) as f:
    STATIONS = json.load(f)
with open(os.path.join(BASE, "data", "network", "tracks.json")) as f:
    TRACKS = json.load(f)
with open(os.path.join(BASE, "data", "network", "trains.json")) as f:
    TRAINS = json.load(f)

STATION_IDS = [s["id"] for s in STATIONS]
ADJACENCY = {}  # station_id -> list of neighbour station_ids
for t in TRACKS:
    ADJACENCY.setdefault(t["from_id"], []).append(t["to_id"])
    ADJACENCY.setdefault(t["to_id"], []).append(t["from_id"])  # bidirectional

# ── Helper: realistic delay distribution ─────────────────────────────────────

def base_delay(hour, month, is_junction):
    """Returns a delay in minutes drawn from a realistic distribution."""
    # Rush-hour multiplier (07-10, 17-21)
    rush = 1.6 if hour in range(7, 11) or hour in range(17, 22) else 1.0
    # Monsoon months (June-September)
    monsoon = 1.8 if 6 <= month <= 9 else 1.0
    # Winter fog (December-February, night/morning)
    fog = 2.2 if month in (12, 1, 2) and hour < 9 else 1.0
    # Junction complexity
    junction_m = 1.3 if is_junction else 1.0

    multiplier = rush * monsoon * fog * junction_m

    # Mixture: 65% on-time (<5 min), 25% moderate (5-30), 10% severe (30-180)
    r = random.random()
    if r < 0.65:
        delay = random.gauss(2, 2) * multiplier
    elif r < 0.90:
        delay = random.gauss(15, 8) * multiplier
    else:
        delay = random.gauss(65, 30) * multiplier

    return max(0.0, round(delay, 1))


def cascade_delay(parent_delay, distance_hops):
    """Downstream delay given upstream parent delay."""
    if parent_delay < 5:
        return 0.0
    dampening = 0.55 ** distance_hops
    noise = random.gauss(1.0, 0.15)
    return max(0.0, round(parent_delay * dampening * noise, 1))

# ── Generate delay records ────────────────────────────────────────────────────

JUNCTION_SET = {s["id"] for s in STATIONS if s.get("is_junction")}
START_DATE = datetime(2019, 1, 1, 0, 0)
END_DATE   = datetime(2024, 1, 1, 0, 0)

delay_rows = []   # (timestamp, train_id, station_id, delay_min, cause)
cascade_pairs = []  # (ts, upstream_station, downstream_station, up_delay, down_delay)

CAUSES = ["signal_failure", "loco_fault", "track_work", "passenger_rush",
          "weather", "level_crossing", "crew_change", "fog", "on_time"]

print("Generating 5-year delay dataset …")
current = START_DATE
day_count = 0

while current < END_DATE:
    hour = current.hour
    month = current.month

    for train in TRAINS:
        # Each train visits ~4-6 stations per day (simplified route)
        route = [train["current_station"]]
        visited = {train["current_station"]}
        next_s = train.get("next_station")
        if next_s and next_s not in visited:
            route.append(next_s)
            visited.add(next_s)
        # extend route randomly using adjacency
        for _ in range(random.randint(2, 4)):
            last = route[-1]
            neighbours = [n for n in ADJACENCY.get(last, []) if n not in visited]
            if not neighbours:
                break
            nxt = random.choice(neighbours)
            route.append(nxt)
            visited.add(nxt)

        prev_delay = 0.0
        for hop, station_id in enumerate(route):
            is_junc = station_id in JUNCTION_SET
            if hop == 0:
                d = base_delay(hour, month, is_junc)
                cause_idx = CAUSES.index("on_time") if d < 5 else random.randint(0, 7)
            else:
                d = cascade_delay(prev_delay, 1) + base_delay(hour, month, is_junc) * 0.3
                d = round(d, 1)
                cause_idx = CAUSES.index("on_time") if d < 5 else random.randint(0, 7)

                # Record cascade pair
                if prev_delay > 10:
                    cascade_pairs.append({
                        "timestamp": current.isoformat(),
                        "upstream_station": route[hop - 1],
                        "downstream_station": station_id,
                        "upstream_delay": prev_delay,
                        "downstream_delay": d,
                        "hop_distance": 1,
                    })

            delay_rows.append({
                "timestamp": current.isoformat(),
                "train_id": train["id"],
                "station_id": station_id,
                "delay_min": d,
                "cause": CAUSES[cause_idx],
                "hour": hour,
                "month": month,
                "is_junction": int(is_junc),
                "day_of_week": current.weekday(),
            })
            prev_delay = d

    # Advance by 1 hour
    current += timedelta(hours=1)
    day_count += 1
    if day_count % (24 * 30) == 0:
        print(f"  {current.strftime('%Y-%m')} … {len(delay_rows):,} records so far")

# ── Save ──────────────────────────────────────────────────────────────────────

out_dir = os.path.join(BASE, "data", "historical")
os.makedirs(out_dir, exist_ok=True)

delays_path = os.path.join(out_dir, "delays.csv")
cascade_path = os.path.join(out_dir, "cascade_pairs.csv")

with open(delays_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=delay_rows[0].keys())
    writer.writeheader()
    writer.writerows(delay_rows)

with open(cascade_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cascade_pairs[0].keys())
    writer.writeheader()
    writer.writerows(cascade_pairs)

print(f"\n[OK] Done!")
print(f"   delays.csv      : {len(delay_rows):,} records -> {delays_path}")
print(f"   cascade_pairs.csv: {len(cascade_pairs):,} pairs -> {cascade_path}")
