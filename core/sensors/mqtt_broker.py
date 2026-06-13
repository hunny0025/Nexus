"""NEXUS — MQTT subscriber with rolling buffers per sensor."""

import json
import threading
from collections import defaultdict, deque
from typing import Callable, Optional

import paho.mqtt.client as mqtt

TOPIC_WILDCARD = "nexus/sensors/#"


class MQTTSubscriber:
    """Subscribes to NEXUS sensor topics and routes messages to callbacks.

    Parameters
    ----------
    broker_host : str
        MQTT broker hostname.
    broker_port : int
        MQTT broker port.
    buffer_size : int
        Maximum number of readings to keep per sensor_id (default 100).
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        buffer_size: int = 100,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.buffer_size = buffer_size

        # Rolling buffers: sensor_id → deque of raw float values
        self._buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.buffer_size)
        )

        # Full message buffers: sensor_id → deque of full message dicts
        self._message_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.buffer_size)
        )

        # Registered callbacks: list of callables
        self._callbacks: list[Callable] = []

        # MQTT client
        self._client = mqtt.Client(
            client_id="nexus-mqtt-subscriber",
            protocol=mqtt.MQTTv311,
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_callback(
        self, callback: Callable[[str, float, str, str], None]
    ):
        """Register a callback invoked for every received message.

        Parameters
        ----------
        callback : Callable
            Signature: callback(sensor_id, value, location, timestamp)
        """
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # MQTT handlers
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(TOPIC_WILDCARD, qos=0)
            print(f"[MQTT] Subscribed to {TOPIC_WILDCARD}")
        else:
            print(f"MQTT connection failed with code {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            sensor_id = payload["sensor_id"]
            value = float(payload["value"])
            location = payload["location"]
            timestamp = payload["timestamp"]

            with self._lock:
                self._buffers[sensor_id].append(value)
                self._message_buffers[sensor_id].append(payload)

            # Dispatch to all registered callbacks
            for cb in self._callbacks:
                try:
                    cb(sensor_id, value, location, timestamp)
                except Exception as exc:
                    print(f"Callback error: {exc}")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"Bad MQTT message on {msg.topic}: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_recent(self, sensor_id: str, n: int = 30) -> list[float]:
        """Return the last *n* readings for a sensor as a list of floats."""
        with self._lock:
            buf = self._buffers.get(sensor_id, deque())
            items = list(buf)
        return items[-n:]

    def get_recent_messages(self, sensor_id: str, n: int = 30) -> list[dict]:
        """Return the last *n* full message dicts for a sensor."""
        with self._lock:
            buf = self._message_buffers.get(sensor_id, deque())
            items = list(buf)
        return items[-n:]

    def get_all_sensor_ids(self) -> list[str]:
        """Return all sensor IDs seen so far."""
        with self._lock:
            return list(self._buffers.keys())

    def start(self, blocking: bool = True):
        """Connect and start listening.

        Parameters
        ----------
        blocking : bool
            If True, blocks the current thread. If False, runs the
            network loop in a background thread.
        """
        self._client.connect(self.broker_host, self.broker_port, keepalive=60)
        if blocking:
            self._client.loop_forever()
        else:
            self._client.loop_start()

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
