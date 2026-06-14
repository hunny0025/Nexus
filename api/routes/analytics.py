"""NEXUS — Analytics API routes."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/learning-curve")
async def get_learning_curve(request: Request):
    """Return accuracy history from learning agent."""
    learning_agent = getattr(request.app.state, "learning_agent", None)
    if learning_agent is None:
        return {"history": []}

    return {"history": learning_agent.get_accuracy_history()}


@router.get("/system-stats")
async def get_system_stats(request: Request):
    """Return aggregate system statistics."""
    learning_agent = getattr(request.app.state, "learning_agent", None)
    escalator = getattr(request.app.state, "escalator", None)

    total_incidents = 0
    autonomous_count = 0
    total_confidence = 0.0
    total_delay_reduction = 0.0

    if learning_agent is not None:
        records = learning_agent._records
        total_incidents = len(records)

        for rec in records.values():
            outcome = rec.get("predicted_intervention_outcome", {})
            if outcome:
                autonomous_count += 1
                total_confidence += outcome.get("confidence", 0.5)
                # Use the predicted delay reduction from the AI (predictive KPI)
                intervention_obj = outcome.get("intervention", {})
                reduction = intervention_obj.get("estimated_delay_reduction_pct", 0.0)
                total_delay_reduction += reduction

    autonomous_rate = (
        (autonomous_count / total_incidents * 100) if total_incidents > 0 else 0
    )
    avg_confidence = (
        total_confidence / autonomous_count if autonomous_count > 0 else 0
    )
    avg_delay_reduction = (
        (total_delay_reduction / autonomous_count * 100)
        if autonomous_count > 0
        else 0
    )

    pending_escalations = 0
    if escalator is not None:
        pending_escalations = len(escalator.get_pending())

    return {
        "total_incidents_handled": total_incidents,
        "autonomous_rate_pct": round(autonomous_rate, 1),
        "avg_confidence": round(avg_confidence, 3),
        "avg_delay_reduction_pct": round(avg_delay_reduction, 1),
        "pending_escalations": pending_escalations,
    }
