"""NEXUS — Integration test for sensor simulator + MQTT subscriber + Kalman."""

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sensors.kalman import KalmanFilterBank
from core.sensors.mqtt_broker import MQTTSubscriber
from core.sensors.simulator import SensorSimulator

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "network"


def load_ids():
    with open(DATA_DIR / "tracks.json", "r") as f:
        track_ids = [t["id"] for t in json.load(f)]
    with open(DATA_DIR / "trains.json", "r") as f:
        train_ids = [t["id"] for t in json.load(f)]
    return track_ids, train_ids


def main():
    track_ids, train_ids = load_ids()
    broker = os.getenv("MQTT_BROKER", "localhost")

    # --- Kalman bank ---
    kalman_bank = KalmanFilterBank(anomaly_threshold=3.0)

    # --- Callback: print + validate ---
    reading_count = {"n": 0}
    readings_log = []

    def on_reading(sensor_id, value, location, timestamp):
        result = kalman_bank.validate_reading(sensor_id, value)
        readings_log.append(result)
        reading_count["n"] += 1
        anomaly_flag = " [!] ANOMALY" if result["is_genuine_anomaly"] else ""
        print(
            f"  [{reading_count['n']:3d}] {sensor_id:35s}  "
            f"raw={value:8.3f}  est={result['estimated_true_value']:8.3f}  "
            f"z={result['z_score']:5.2f}{anomaly_flag}"
        )

    # --- Subscriber ---
    subscriber = MQTTSubscriber(broker_host=broker)
    subscriber.register_callback(on_reading)
    subscriber.start(blocking=False)

    # --- Simulator ---
    sim = SensorSimulator(
        broker_host=broker,
        track_ids=track_ids[:3],   # Use 3 tracks for brevity
        train_ids=train_ids[:2],   # Use 2 trains for brevity
        interval_sec=0.5,
    )
    sim_thread = threading.Thread(target=sim.start, daemon=True)
    sim_thread.start()

    # --- Phase 1: Normal readings ---
    print("\n" + "=" * 80)
    print("PHASE 1 - Normal sensor readings (collecting ~10 readings)")
    print("=" * 80)
    target = 10
    timeout = time.time() + 15
    while reading_count["n"] < target and time.time() < timeout:
        time.sleep(0.3)

    # --- Phase 2: Inject fault ---
    fault_location = track_ids[0]
    print(f"\n{'=' * 80}")
    print(f"PHASE 2 - Injecting fault at {fault_location}")
    print("=" * 80)
    sim.inject_fault(fault_location, ["vibration", "track_stress"])

    # Reset counter to count post-fault readings
    pre_fault_count = reading_count["n"]
    target_post = pre_fault_count + 10
    timeout = time.time() + 20
    while reading_count["n"] < target_post and time.time() < timeout:
        time.sleep(0.3)

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print("=" * 80)
    total = reading_count["n"]
    anomalies = sum(1 for r in readings_log if r["is_genuine_anomaly"])
    print(f"  Total readings processed : {total}")
    print(f"  Anomalies detected       : {anomalies}")
    print(f"  Anomaly rate             : {anomalies / max(total, 1) * 100:.1f}%")

    if anomalies > 0:
        print("\n[OK] Fault injection -> anomaly detection pipeline WORKING")
    else:
        print("\n[WARN] No anomalies detected - fault may need more cycles to ramp up")

    # Cleanup
    sim.stop()
    subscriber.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
