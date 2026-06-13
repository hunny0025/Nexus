"""NEXUS — Real-time sensor simulator publishing to MQTT."""

import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Sensor baseline distributions (mean, std)
# ---------------------------------------------------------------------------

TRACK_SENSOR_BASELINES = {
    "vibration": (0.5, 0.08),
    "track_stress": (100.0, 12.0),
}

TRAIN_SENSOR_BASELINES = {
    "temperature": (45.0, 3.0),
    "brake_pressure": (6.0, 0.4),
    "wheel_impact": (1.2, 0.15),
}

# Fault multipliers applied to means when a fault is injected
FAULT_MULTIPLIERS = {
    "vibration": 4.0,
    "track_stress": 2.5,
    "temperature": 2.0,
    "brake_pressure": 0.3,    # drops on failure
    "wheel_impact": 5.0,
}

SEVERITY_RAMP = [0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0]

TOPIC_PREFIX = "nexus/sensors"


class SensorSimulator:
    """Publishes synthetic sensor readings to an MQTT broker.

    Parameters
    ----------
    broker_host : str
        MQTT broker hostname.
    broker_port : int
        MQTT broker port.
    track_ids : list[str]
        List of TrackSection IDs to simulate.
    train_ids : list[str]
        List of Train IDs to simulate.
    interval_sec : float
        Seconds between publish cycles (default 2).
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        track_ids: list | None = None,
        train_ids: list | None = None,
        interval_sec: float = 2.0,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.track_ids = track_ids or []
        self.train_ids = train_ids or []
        self.interval_sec = interval_sec

        # {location_id: {sensor_type: cycle_count}} — tracks fault progression
        self._active_faults: dict[str, dict[str, int]] = {}

        # MQTT client
        self._client = mqtt.Client(
            client_id=f"nexus-sensor-sim-{random.randint(1000,9999)}",
            protocol=mqtt.MQTTv311,
        )
        self._running = False

    # ------------------------------------------------------------------
    # MQTT lifecycle
    # ------------------------------------------------------------------

    def _connect(self):
        self._client.connect(self.broker_host, self.broker_port, keepalive=60)
        self._client.loop_start()

    def _disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    # ------------------------------------------------------------------
    # Reading generation
    # ------------------------------------------------------------------

    def _fault_severity(self, location_id: str, sensor_type: str) -> float:
        """Return current severity multiplier (0.0 if no fault)."""
        if location_id not in self._active_faults:
            return 0.0
        if sensor_type not in self._active_faults[location_id]:
            return 0.0
        cycle = self._active_faults[location_id][sensor_type]
        idx = min(cycle, len(SEVERITY_RAMP) - 1)
        return SEVERITY_RAMP[idx]

    def _advance_fault_cycle(self, location_id: str, sensor_type: str):
        if location_id in self._active_faults:
            if sensor_type in self._active_faults[location_id]:
                self._active_faults[location_id][sensor_type] += 1

    def _generate_reading(
        self, location_id: str, sensor_type: str, mean: float, std: float
    ) -> float:
        """Generate a single sensor reading, applying fault shift if active."""
        base_value = random.gauss(mean, std)
        severity = self._fault_severity(location_id, sensor_type)
        if severity > 0.0:
            multiplier = FAULT_MULTIPLIERS.get(sensor_type, 1.0)
            # For brake_pressure the multiplier is <1, so shift is negative
            if multiplier < 1.0:
                fault_shift = mean * (1.0 - multiplier) * severity * -1
            else:
                fault_shift = mean * (multiplier - 1.0) * severity
            base_value += fault_shift
            self._advance_fault_cycle(location_id, sensor_type)
        return round(base_value, 4)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_reading(
        self, location_id: str, sensor_type: str, value: float
    ):
        sensor_id = f"{location_id}_{sensor_type}"
        topic = f"{TOPIC_PREFIX}/{location_id}/{sensor_type}"
        payload = json.dumps(
            {
                "sensor_id": sensor_id,
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location": location_id,
            }
        )
        self._client.publish(topic, payload, qos=0)

    def _publish_cycle(self):
        """Publish one round of readings for all locations."""
        # Track sensors
        for track_id in self.track_ids:
            for sensor_type, (mean, std) in TRACK_SENSOR_BASELINES.items():
                value = self._generate_reading(track_id, sensor_type, mean, std)
                self._publish_reading(track_id, sensor_type, value)

        # Train sensors
        for train_id in self.train_ids:
            for sensor_type, (mean, std) in TRAIN_SENSOR_BASELINES.items():
                value = self._generate_reading(train_id, sensor_type, mean, std)
                self._publish_reading(train_id, sensor_type, value)

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------

    def inject_fault(self, location_id: str, sensor_types: list[str]):
        """Begin injecting faults for the given sensors at a location.

        Severity ramps from 0.3 → 1.0 over 10 publish cycles.
        """
        if location_id not in self._active_faults:
            self._active_faults[location_id] = {}
        for st in sensor_types:
            self._active_faults[location_id][st] = 0
        print(
            f"[FAULT] Injected at {location_id} on sensors: {sensor_types}"
        )

    def clear_fault(self, location_id: str):
        """Remove all active faults for a location."""
        self._active_faults.pop(location_id, None)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self):
        """Connect and begin publishing sensor readings in a loop."""
        self._connect()
        self._running = True
        print(
            f"[SIM] Sensor simulator started - publishing every {self.interval_sec}s "
            f"for {len(self.track_ids)} tracks, {len(self.train_ids)} trains"
        )
        try:
            while self._running:
                self._publish_cycle()
                time.sleep(self.interval_sec)
        except KeyboardInterrupt:
            pass
        finally:
            self._disconnect()
            print("Sensor simulator stopped.")

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "network"

    with open(DATA_DIR / "tracks.json", "r") as f:
        track_ids = [t["id"] for t in json.load(f)]
    with open(DATA_DIR / "trains.json", "r") as f:
        train_ids = [t["id"] for t in json.load(f)]

    broker = os.getenv("MQTT_BROKER", "localhost")
    sim = SensorSimulator(
        broker_host=broker, track_ids=track_ids, train_ids=train_ids
    )
    sim.start()
