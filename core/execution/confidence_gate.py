"""NEXUS — Confidence gate for autonomous vs escalated decision-making."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GateDecision:
    """Result of the confidence gate evaluation."""

    action: str  # "AUTONOMOUS" or "ESCALATE"
    confidence: float
    reason: str
    selected_intervention: dict
    alternatives: list = field(default_factory=list)


class ConfidenceGate:
    """Evaluates whether NEXUS should act autonomously or escalate.

    Parameters
    ----------
    autonomous_threshold : float
        Minimum confidence to act without human approval (default 0.75).
    """

    def __init__(self, autonomous_threshold: float = 0.75):
        self.autonomous_threshold = autonomous_threshold

    def evaluate(
        self,
        pareto_results: list[dict],
        cascade_map: dict[str, dict[str, float]],
        incident_history_match_score: float = 0.0,
    ) -> GateDecision:
        """Evaluate the confidence gate.

        Parameters
        ----------
        pareto_results : list[dict]
            Pareto-ranked interventions from the optimizer.
        cascade_map : dict
            Cascade probability map from propagation engine.
        incident_history_match_score : float
            How well this incident matches historical patterns (0–1).

        Returns
        -------
        GateDecision
        """
        if not pareto_results:
            return GateDecision(
                action="ESCALATE",
                confidence=0.0,
                reason="No interventions available",
                selected_intervention={},
                alternatives=[],
            )

        top = pareto_results[0]
        sim_confidence = top.get("confidence", 0.0)
        history_confidence = incident_history_match_score
        cascade_confidence = self._compute_cascade_confidence(cascade_map)

        # Weighted average
        confidence = (
            sim_confidence * 0.40
            + history_confidence * 0.35
            + cascade_confidence * 0.25
        )
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        # Decision
        if confidence >= self.autonomous_threshold:
            action = "AUTONOMOUS"
            reason = (
                f"Confidence {confidence:.2f} >= threshold "
                f"{self.autonomous_threshold:.2f} "
                f"(sim={sim_confidence:.2f}, history={history_confidence:.2f}, "
                f"cascade={cascade_confidence:.2f})"
            )
        else:
            action = "ESCALATE"
            reason = (
                f"Confidence {confidence:.2f} < threshold "
                f"{self.autonomous_threshold:.2f} — requires human approval "
                f"(sim={sim_confidence:.2f}, history={history_confidence:.2f}, "
                f"cascade={cascade_confidence:.2f})"
            )

        return GateDecision(
            action=action,
            confidence=confidence,
            reason=reason,
            selected_intervention=top.get("intervention", {}),
            alternatives=[r.get("intervention", {}) for r in pareto_results[1:4]],
        )

    def _compute_cascade_confidence(
        self, cascade_map: dict[str, dict[str, float]]
    ) -> float:
        """Cascade confidence: lower if spread is high and variance is large."""
        if not cascade_map:
            return 0.5

        max_probs = []
        time_variances = []
        for node, time_probs in cascade_map.items():
            values = list(time_probs.values())
            if not values:
                continue
            max_probs.append(max(values))
            if len(values) > 1:
                mean_v = sum(values) / len(values)
                variance = sum((v - mean_v) ** 2 for v in values) / len(values)
                time_variances.append(variance)

        if not max_probs:
            return 0.5

        # Higher max probabilities → lower confidence (worse situation)
        avg_max = sum(max_probs) / len(max_probs)
        # Higher variance → less predictable → lower confidence
        avg_var = sum(time_variances) / len(time_variances) if time_variances else 0.0

        confidence = max(0.0, 1.0 - avg_max - avg_var * 2)
        return round(confidence, 4)
