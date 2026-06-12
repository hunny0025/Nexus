"""NEXUS — Kalman filter for sensor validation and anomaly scoring."""

import math
from collections import defaultdict


class SensorKalmanFilter:
    """Scalar Kalman filter for a single sensor stream.

    Validates incoming readings against a predicted state. Returns
    z-scores indicating how far a reading deviates from the filter's
    expectation.

    Parameters
    ----------
    process_variance : float
        Q — how much we expect the true value to change between steps.
    measurement_variance : float
        R — how noisy the sensor readings are.
    initial_estimate : float
        x̂₀ — starting state estimate.
    initial_error : float
        P₀ — starting estimate covariance.
    """

    def __init__(
        self,
        process_variance: float = 1e-3,
        measurement_variance: float = 0.05,
        initial_estimate: float = 0.0,
        initial_error: float = 1.0,
    ):
        self.Q = process_variance
        self.R = measurement_variance
        self.x_hat = initial_estimate  # state estimate
        self.P = initial_error          # estimate covariance

    def update(self, measurement: float) -> dict:
        """Run one predict–update cycle and return validation result.

        Returns
        -------
        dict
            Keys: predicted, residual, z_score, estimated, kalman_gain
        """
        # --- Predict ---
        x_pred = self.x_hat
        P_pred = self.P + self.Q

        # --- Update ---
        residual = measurement - x_pred
        S = P_pred + self.R  # innovation covariance
        K = P_pred / S       # Kalman gain

        self.x_hat = x_pred + K * residual
        self.P = (1 - K) * P_pred

        # z-score: how many standard deviations is the residual?
        z_score = abs(residual) / math.sqrt(S) if S > 0 else 0.0

        return {
            "predicted": round(x_pred, 6),
            "residual": round(residual, 6),
            "z_score": round(z_score, 4),
            "estimated": round(self.x_hat, 6),
            "kalman_gain": round(K, 6),
        }

    def validate_reading(
        self, sensor_id: str, raw_value: float, anomaly_threshold: float = 3.0
    ) -> dict:
        """Convenience wrapper that returns the full validation dict.

        Returns
        -------
        dict
            sensor_id, raw_value, estimated_true_value, z_score,
            is_genuine_anomaly
        """
        result = self.update(raw_value)
        return {
            "sensor_id": sensor_id,
            "raw_value": round(raw_value, 4),
            "estimated_true_value": result["estimated"],
            "z_score": result["z_score"],
            "is_genuine_anomaly": result["z_score"] > anomaly_threshold,
        }


class KalmanFilterBank:
    """Manages one SensorKalmanFilter per sensor_id.

    Automatically creates filters on first sight using default parameters,
    or accepts per-sensor configuration via ``configure()``.
    """

    def __init__(
        self,
        default_process_variance: float = 1e-3,
        default_measurement_variance: float = 0.05,
        anomaly_threshold: float = 3.0,
    ):
        self.default_Q = default_process_variance
        self.default_R = default_measurement_variance
        self.anomaly_threshold = anomaly_threshold

        # sensor_id → SensorKalmanFilter
        self._filters: dict[str, SensorKalmanFilter] = {}

        # Optional per-sensor config overrides: sensor_id → (Q, R)
        self._configs: dict[str, tuple[float, float]] = {}

    def configure(
        self, sensor_id: str, process_variance: float, measurement_variance: float
    ):
        """Set custom Kalman parameters for a specific sensor."""
        self._configs[sensor_id] = (process_variance, measurement_variance)
        # Reset filter if it already exists
        self._filters.pop(sensor_id, None)

    def _get_or_create_filter(
        self, sensor_id: str, initial_value: float = 0.0
    ) -> SensorKalmanFilter:
        if sensor_id not in self._filters:
            Q, R = self._configs.get(sensor_id, (self.default_Q, self.default_R))
            self._filters[sensor_id] = SensorKalmanFilter(
                process_variance=Q,
                measurement_variance=R,
                initial_estimate=initial_value,
                initial_error=1.0,
            )
        return self._filters[sensor_id]

    def validate_reading(self, sensor_id: str, raw_value: float) -> dict:
        """Validate a single sensor reading through its Kalman filter.

        Returns
        -------
        dict
            sensor_id, raw_value, estimated_true_value, z_score,
            is_genuine_anomaly
        """
        kf = self._get_or_create_filter(sensor_id, initial_value=raw_value)
        return kf.validate_reading(
            sensor_id, raw_value, anomaly_threshold=self.anomaly_threshold
        )

    def validate_batch(
        self, readings: dict[str, float]
    ) -> dict[str, dict]:
        """Validate multiple sensor readings at once.

        Parameters
        ----------
        readings : dict
            Mapping of sensor_id → raw_value.

        Returns
        -------
        dict
            Mapping of sensor_id → validation result dict.
        """
        results = {}
        for sensor_id, value in readings.items():
            results[sensor_id] = self.validate_reading(sensor_id, value)
        return results

    def get_anomalous_sensors(
        self, readings: dict[str, float]
    ) -> list[dict]:
        """Return only the sensors flagged as genuine anomalies."""
        batch = self.validate_batch(readings)
        return [v for v in batch.values() if v["is_genuine_anomaly"]]

    def reset(self, sensor_id: str | None = None):
        """Reset one or all filters."""
        if sensor_id is None:
            self._filters.clear()
        else:
            self._filters.pop(sensor_id, None)
