"""NEXUS — Multi-source anomaly fusion engine."""

from typing import Optional

import numpy as np
import torch

from core.detection.lstm_detector import AnomalyDetectionPipeline
from core.detection.pattern_detector import GraphPatternDetector
from core.sensors.kalman import KalmanFilterBank


class AnomalyFusionEngine:
    """Fuses LSTM, Kalman, and graph-pattern evidence to confirm anomalies.

    A confirmed anomaly requires:
    1. The LSTM pipeline flags the location as anomalous.
    2. At least 2 individual sensors have Kalman z-score > 2.5.

    Parameters
    ----------
    lstm_pipeline : AnomalyDetectionPipeline
        Trained LSTM sliding-window pipeline.
    kalman_bank : KalmanFilterBank
        Per-sensor Kalman filter bank.
    pattern_detector : GraphPatternDetector
        Neo4j graph pattern detector.
    kalman_z_threshold : float
        Per-sensor z-score threshold for the Kalman "hit" count (default 2.5).
    min_kalman_hits : int
        Minimum sensors exceeding z-threshold to co-confirm (default 2).
    """

    def __init__(
        self,
        lstm_pipeline: AnomalyDetectionPipeline,
        kalman_bank: KalmanFilterBank,
        pattern_detector: GraphPatternDetector,
        kalman_z_threshold: float = 2.5,
        min_kalman_hits: int = 2,
    ):
        self.lstm_pipeline = lstm_pipeline
        self.kalman_bank = kalman_bank
        self.pattern_detector = pattern_detector
        self.kalman_z_threshold = kalman_z_threshold
        self.min_kalman_hits = min_kalman_hits

    def fuse(
        self,
        location_id: str,
        recent_readings: dict[str, float],
    ) -> dict:
        """Run all three detection engines and fuse results.

        Parameters
        ----------
        location_id : str
            The track or train identifier.
        recent_readings : dict[str, float]
            Mapping of sensor_id → latest raw value.
            E.g. {"TRK_NDLS_CNB_vibration": 2.1, ...}

        Returns
        -------
        dict
            {confirmed, location_id, confidence, evidence}
        """
        evidence: dict = {
            "lstm_score": None,
            "lstm_anomalous": False,
            "kalman_hits": 0,
            "kalman_details": [],
            "patterns_matched": [],
        }

        # -----------------------------------------------------------
        # 1. LSTM scoring
        # -----------------------------------------------------------
        lstm_result = self.lstm_pipeline.score_location(location_id)
        if lstm_result is not None:
            evidence["lstm_score"] = lstm_result["anomaly_score"]
            evidence["lstm_anomalous"] = lstm_result["is_anomalous"]
        else:
            # Window not full — cannot score yet
            evidence["lstm_score"] = 0.0
            evidence["lstm_anomalous"] = False

        # -----------------------------------------------------------
        # 2. Kalman validation on each sensor
        # -----------------------------------------------------------
        kalman_hits = 0
        for sensor_id, raw_value in recent_readings.items():
            result = self.kalman_bank.validate_reading(sensor_id, raw_value)
            if result["z_score"] > self.kalman_z_threshold:
                kalman_hits += 1
            evidence["kalman_details"].append(result)

        evidence["kalman_hits"] = kalman_hits

        # -----------------------------------------------------------
        # 3. Graph pattern detection
        # -----------------------------------------------------------
        try:
            patterns = self.pattern_detector.detect_precursor_patterns()
            # Filter to patterns related to this location
            matched = [
                p
                for p in patterns
                if p["location_id"] == location_id
                or location_id in p.get("description", "")
            ]
            evidence["patterns_matched"] = matched
        except Exception:
            evidence["patterns_matched"] = []

        # -----------------------------------------------------------
        # Fusion decision: confirm if we have enough Kalman hits.
        # This acts as an immediate fallback when the LSTM window is filling up or untrained.
        # -----------------------------------------------------------
        confirmed = (kalman_hits >= self.min_kalman_hits) or (
            evidence["lstm_anomalous"] and kalman_hits >= self.min_kalman_hits
        )
        if confirmed or kalman_hits > 0:
            print(f"[DEBUG FUSE] location={location_id} kalman_hits={kalman_hits} min_hits={self.min_kalman_hits} lstm_anom={evidence['lstm_anomalous']} details={[{'sid': d['sensor_id'], 'z': d['z_score']} for d in evidence['kalman_details']]}")

        # Confidence: weighted combination of evidence strength
        confidence = 0.0
        if confirmed:
            lstm_conf = min(1.0, evidence["lstm_score"] / max(self.lstm_pipeline.model.threshold, 1e-9)) * 0.5
            kalman_conf = min(1.0, kalman_hits / max(len(recent_readings), 1)) * 0.3
            pattern_conf = min(1.0, len(evidence["patterns_matched"]) * 0.25) * 0.2
            confidence = lstm_conf + kalman_conf + pattern_conf
        elif evidence["lstm_anomalous"] or kalman_hits >= self.min_kalman_hits:
            # Partial evidence
            confidence = 0.3

        return {
            "confirmed": confirmed,
            "location_id": location_id,
            "confidence": round(min(confidence, 1.0), 4),
            "evidence": evidence,
        }
