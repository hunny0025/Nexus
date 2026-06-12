"""NEXUS — Graph sync agent: keeps Neo4j in sync with live sensor data."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GraphSyncAgent:
    """Periodically syncs MQTT sensor readings into Neo4j graph properties.

    Parameters
    ----------
    neo4j_driver : neo4j.Driver
        Neo4j driver.
    mqtt_subscriber : MQTTSubscriber
        The MQTT subscriber with buffered readings.
    kalman_bank : KalmanFilterBank or None
        Optional Kalman filter bank for validated readings.
    """

    def __init__(self, neo4j_driver, mqtt_subscriber, kalman_bank=None):
        self.driver = neo4j_driver
        self.subscriber = mqtt_subscriber
        self.kalman_bank = kalman_bank
        self._running = False

    async def sync_loop(self, interval: int = 10):
        """Run sync every *interval* seconds.

        Parameters
        ----------
        interval : int
            Seconds between sync cycles (default 10).
        """
        self._running = True
        logger.info(f"Graph sync agent started (interval={interval}s)")

        while self._running:
            try:
                await self._sync_cycle()
            except Exception as exc:
                logger.error(f"Sync cycle error: {exc}")
            await asyncio.sleep(interval)

    async def _sync_cycle(self):
        """One sync cycle: read buffers → update Neo4j."""
        sensor_ids = self.subscriber.get_all_sensor_ids()
        if not sensor_ids:
            return

        # Group by location
        location_readings: dict[str, dict[str, float]] = {}
        for sensor_id in sensor_ids:
            recent = self.subscriber.get_recent(sensor_id, n=5)
            if not recent:
                continue

            avg_value = sum(recent) / len(recent)

            # Parse location from sensor_id (format: location_sensortype)
            parts = sensor_id.rsplit("_", 1)
            if len(parts) != 2:
                continue
            location = parts[0]
            sensor_type = parts[1]

            if location not in location_readings:
                location_readings[location] = {}
            location_readings[location][sensor_type] = avg_value

        # Update Neo4j
        for location, readings in location_readings.items():
            if location.startswith("TRK_"):
                await asyncio.to_thread(
                    self.update_track_health, location, readings
                )
            else:
                # Train or station update
                await asyncio.to_thread(
                    self._update_train_health, location, readings
                )

    def update_track_health(self, track_id: str, sensor_readings: dict[str, float]):
        """Update TRACK relationship health probabilities based on readings.

        Parameters
        ----------
        track_id : str
            TrackSection ID.
        sensor_readings : dict
            {sensor_type: averaged_value}
        """
        vibration = sensor_readings.get("vibration", 0.5)
        stress = sensor_readings.get("stress", 100.0)

        # Derive health probabilities from sensor values
        # Higher vibration / stress → more likely degraded/failed
        vib_ratio = vibration / 0.5  # ratio to baseline
        stress_ratio = stress / 100.0

        # Simple health model
        if vib_ratio > 3.0 or stress_ratio > 2.0:
            p_healthy = 0.10
            p_degraded = 0.30
            p_failed = 0.60
        elif vib_ratio > 2.0 or stress_ratio > 1.5:
            p_healthy = 0.40
            p_degraded = 0.45
            p_failed = 0.15
        elif vib_ratio > 1.5:
            p_healthy = 0.70
            p_degraded = 0.25
            p_failed = 0.05
        else:
            p_healthy = 0.95
            p_degraded = 0.04
            p_failed = 0.01

        query = """
        MATCH ()-[r:TRACK {id: $track_id}]->()
        SET r.health_prob_healthy = $p_healthy,
            r.health_prob_degraded = $p_degraded,
            r.health_prob_failed = $p_failed,
            r.last_sensor_update = datetime()
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    track_id=track_id,
                    p_healthy=round(p_healthy, 4),
                    p_degraded=round(p_degraded, 4),
                    p_failed=round(p_failed, 4),
                )
        except Exception as exc:
            logger.error(f"Track health update failed for {track_id}: {exc}")

    def _update_train_health(self, train_id: str, sensor_readings: dict[str, float]):
        """Update train properties based on sensor readings."""
        brake = sensor_readings.get("pressure", 6.0)
        temp = sensor_readings.get("temperature", 45.0)

        status = "ON_TIME"
        if brake < 3.0 or temp > 80.0:
            status = "ALERT"
        elif brake < 5.0 or temp > 60.0:
            status = "WARNING"

        query = """
        MATCH (t:Train {id: $train_id})
        SET t.last_brake_pressure = $brake,
            t.last_temperature = $temp,
            t.sensor_status = $status,
            t.last_sensor_update = datetime()
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    train_id=train_id,
                    brake=round(brake, 2),
                    temp=round(temp, 2),
                    status=status,
                )
        except Exception as exc:
            logger.error(f"Train health update failed for {train_id}: {exc}")

    def stop(self):
        self._running = False
