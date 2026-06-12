"""NEXUS — Escalation handler for human-in-the-loop decisions."""

import uuid
from datetime import datetime, timezone
from typing import Optional


class EscalationHandler:
    """Builds structured briefing packets for human operators when
    the confidence gate escalates a decision.
    """

    def __init__(self):
        # In-memory store of pending escalations
        self._pending: dict[str, dict] = {}

    async def build_brief(
        self,
        gate_decision,
        explanation: str = "",
        full_state: Optional[dict] = None,
    ) -> dict:
        """Build a structured briefing for a human operator.

        Parameters
        ----------
        gate_decision : GateDecision
            The escalated decision.
        explanation : str
            AI-generated explanation text.
        full_state : dict or None
            Current system state snapshot.

        Returns
        -------
        dict
            Structured briefing packet.
        """
        incident_id = f"ESC_{uuid.uuid4().hex[:8].upper()}"
        state = full_state or {}

        # Determine urgency
        cascade_prob = state.get("cascade_probability", 0.5)
        trains_affected = len(state.get("trains_affected", []))
        if cascade_prob > 0.7 or trains_affected > 5:
            urgency = "CRITICAL"
        else:
            urgency = "HIGH"

        # Build top-3 options
        top_3 = []
        selected = gate_decision.selected_intervention
        if selected:
            top_3.append(
                {
                    "intervention": selected,
                    "projected_outcome": {
                        "delay_reduction_pct": selected.get(
                            "estimated_delay_reduction_pct", 0.3
                        ),
                        "cascade_reduction": 0.35
                        if selected.get("type") == "REROUTE"
                        else 0.20,
                    },
                    "confidence": gate_decision.confidence,
                }
            )

        for alt in gate_decision.alternatives[:2]:
            top_3.append(
                {
                    "intervention": alt,
                    "projected_outcome": {
                        "delay_reduction_pct": alt.get(
                            "estimated_delay_reduction_pct", 0.2
                        ),
                        "cascade_reduction": 0.20,
                    },
                    "confidence": gate_decision.confidence * 0.85,
                }
            )

        # Cascade summary
        cascade_summary = state.get(
            "cascade_summary",
            f"Cascade probability {cascade_prob:.0%} affecting "
            f"{trains_affected} trains",
        )

        # Time window
        time_window = 15 if urgency == "CRITICAL" else 30

        brief = {
            "incident_id": incident_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "urgency": urgency,
            "cascade_summary": cascade_summary,
            "top_3_options": top_3,
            "recommended_option": top_3[0] if top_3 else None,
            "time_window_minutes": time_window,
            "explanation_text": explanation,
            "gate_reason": gate_decision.reason,
            "confidence": gate_decision.confidence,
        }

        self._pending[incident_id] = brief
        return brief

    def get_pending(self) -> list[dict]:
        """Return all pending escalations."""
        return list(self._pending.values())

    def resolve(self, incident_id: str, approved_intervention: Optional[dict] = None):
        """Mark an escalation as resolved."""
        if incident_id in self._pending:
            self._pending[incident_id]["status"] = "RESOLVED"
            if approved_intervention:
                self._pending[incident_id]["approved"] = approved_intervention
            return self._pending.pop(incident_id, None)
        return None
