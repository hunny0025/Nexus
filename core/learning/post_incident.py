"""NEXUS — Post-incident learning agent for continuous improvement."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
LEARNING_LOG_PATH = DATA_DIR / "learning_log.json"


class LearningAgent:
    """Records predictions vs actuals and adapts model parameters.

    Stores all data in data/learning_log.json (append mode).
    """

    def __init__(self):
        # In-memory records: incident_id → record dict
        self._records: dict[str, dict] = {}
        self._load_existing()

    def _load_existing(self):
        """Load existing records from disk."""
        if LEARNING_LOG_PATH.exists():
            try:
                with open(LEARNING_LOG_PATH, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for rec in records:
                    self._records[rec["incident_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                self._records = {}

    def _save(self):
        """Persist all records to disk."""
        LEARNING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEARNING_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(list(self._records.values()), f, indent=2)

    def record_prediction(
        self,
        incident_id: str,
        predicted_cascade_map: dict,
        predicted_intervention_outcome: dict,
    ):
        """Record the system's predictions for an incident.

        Parameters
        ----------
        incident_id : str
        predicted_cascade_map : dict
            The cascade propagation map.
        predicted_intervention_outcome : dict
            The simulated intervention outcome.
        """
        if incident_id not in self._records:
            self._records[incident_id] = {
                "incident_id": incident_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        self._records[incident_id]["predicted_cascade_map"] = predicted_cascade_map
        self._records[incident_id][
            "predicted_intervention_outcome"
        ] = predicted_intervention_outcome
        self._save()
        logger.info(f"Recorded prediction for {incident_id}")

    def record_actual(
        self,
        incident_id: str,
        actual_trains_affected: list[str],
        actual_delay_minutes: float,
    ):
        """Record the actual outcome after an incident.

        Parameters
        ----------
        incident_id : str
        actual_trains_affected : list[str]
            Trains that were actually affected.
        actual_delay_minutes : float
            Actual total delay in minutes.
        """
        if incident_id not in self._records:
            self._records[incident_id] = {
                "incident_id": incident_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        self._records[incident_id]["actual_trains_affected"] = actual_trains_affected
        self._records[incident_id]["actual_delay_minutes"] = actual_delay_minutes
        self._records[incident_id]["actual_recorded_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._save()
        logger.info(f"Recorded actuals for {incident_id}")

    def compute_accuracy(self, incident_id: str) -> Optional[dict]:
        """Compute prediction accuracy for a specific incident.

        Returns
        -------
        dict or None
            {cascade_accuracy, intervention_accuracy}
        """
        rec = self._records.get(incident_id)
        if rec is None:
            return None

        cascade_accuracy = self._compute_cascade_accuracy(rec)
        intervention_accuracy = self._compute_intervention_accuracy(rec)

        accuracy = {
            "cascade_accuracy": round(cascade_accuracy, 4),
            "intervention_accuracy": round(intervention_accuracy, 4),
        }

        self._records[incident_id]["accuracy"] = accuracy
        self._save()
        return accuracy

    def _compute_cascade_accuracy(self, record: dict) -> float:
        """Compare predicted cascade spread vs actual trains affected."""
        predicted_map = record.get("predicted_cascade_map", {})
        actual_trains = record.get("actual_trains_affected", [])

        if not predicted_map or not actual_trains:
            return 0.5  # No data to compare

        # Count predicted high-risk nodes
        predicted_nodes = sum(
            1
            for node, times in predicted_map.items()
            if any(p > 0.3 for p in times.values())
        )

        actual_count = len(actual_trains)
        if actual_count == 0:
            return 1.0 if predicted_nodes == 0 else 0.5

        # Accuracy: closer the predicted count to actual, better
        ratio = min(predicted_nodes, actual_count) / max(predicted_nodes, actual_count)
        return ratio

    def _compute_intervention_accuracy(self, record: dict) -> float:
        """Compare predicted delay reduction vs actual delay."""
        predicted = record.get("predicted_intervention_outcome", {})
        actual_delay = record.get("actual_delay_minutes")

        if not predicted or actual_delay is None:
            return 0.5

        predicted_delay = predicted.get("total_delay_minutes", actual_delay)
        if actual_delay == 0:
            return 1.0 if predicted_delay < 10 else 0.5

        error_ratio = abs(predicted_delay - actual_delay) / max(actual_delay, 1)
        accuracy = max(0.0, 1.0 - error_ratio)
        return accuracy

    def get_accuracy_history(self) -> list[dict]:
        """Return accuracy records for all incidents.

        Returns
        -------
        list[dict]
            Each: {incident_id, timestamp, cascade_accuracy, intervention_accuracy}
        """
        history = []
        for rec in self._records.values():
            accuracy = rec.get("accuracy")
            if accuracy is None:
                accuracy = self.compute_accuracy(rec["incident_id"])
            if accuracy is not None:
                history.append(
                    {
                        "incident_id": rec["incident_id"],
                        "timestamp": rec.get("timestamp", ""),
                        "cascade_accuracy": accuracy.get("cascade_accuracy", 0.0),
                        "intervention_accuracy": accuracy.get(
                            "intervention_accuracy", 0.0
                        ),
                    }
                )
        return history

    def update_dbn_weights(self, belief_engine, incident_id: str):
        """Adapt propagation parameters based on prediction accuracy.

        If cascade_accuracy < 0.6: increase dampening (over-predicted).
        If cascade_accuracy > 0.8: decrease dampening (under-predicted).
        """
        accuracy = self.compute_accuracy(incident_id)
        if accuracy is None:
            return

        cascade_acc = accuracy["cascade_accuracy"]
        current_dampening = getattr(
            belief_engine, "dampening_per_hop", 0.4
        )

        if cascade_acc < 0.6:
            # Over-predicted cascade — reduce propagation factor
            new_dampening = max(0.1, current_dampening - 0.05)
            logger.info(
                f"Learning: cascade accuracy {cascade_acc:.2f} < 0.6 — "
                f"reducing dampening {current_dampening:.2f} → {new_dampening:.2f}"
            )
        elif cascade_acc > 0.8:
            # Under-predicted — increase propagation factor
            new_dampening = min(0.8, current_dampening + 0.03)
            logger.info(
                f"Learning: cascade accuracy {cascade_acc:.2f} > 0.8 — "
                f"increasing dampening {current_dampening:.2f} → {new_dampening:.2f}"
            )
        else:
            new_dampening = current_dampening

        belief_engine.dampening_per_hop = new_dampening

    def get_incident_match_score(self, cascade_summary: str) -> float:
        """Score how well a new incident matches historical patterns.

        Simple heuristic: ratio of past incidents with accuracy > 0.6.
        """
        history = self.get_accuracy_history()
        if not history:
            return 0.3  # Low confidence when no history

        good_matches = sum(
            1
            for h in history
            if h.get("cascade_accuracy", 0) > 0.6
            and h.get("intervention_accuracy", 0) > 0.6
        )
        return min(1.0, good_matches / max(len(history), 1) + 0.2)
