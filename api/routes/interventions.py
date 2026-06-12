"""NEXUS — Intervention approval API routes."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/interventions", tags=["interventions"])


class ApprovalRequest(BaseModel):
    intervention_index: int = 0  # index in top_3_options


@router.get("/pending")
async def get_pending(request: Request):
    """Return all interventions awaiting human approval."""
    escalator = getattr(request.app.state, "escalator", None)
    if escalator is None:
        return {"pending": []}

    return {"pending": escalator.get_pending()}


@router.post("/{incident_id}/approve")
async def approve_intervention(
    request: Request,
    incident_id: str,
    approval: ApprovalRequest,
):
    """Human approves an escalated intervention, triggering executor."""
    escalator = getattr(request.app.state, "escalator", None)
    executor = getattr(request.app.state, "executor", None)
    explainer = getattr(request.app.state, "explainer", None)

    if escalator is None:
        return {"error": "Escalation handler not available"}

    # Find the pending brief
    pending = escalator.get_pending()
    brief = None
    for p in pending:
        if p.get("incident_id") == incident_id:
            brief = p
            break

    if brief is None:
        return {"error": f"No pending escalation for {incident_id}"}

    # Select intervention
    options = brief.get("top_3_options", [])
    idx = min(approval.intervention_index, len(options) - 1)
    if idx < 0 or not options:
        return {"error": "No interventions available"}

    selected = options[idx].get("intervention", {})

    # Execute
    log = {"status": "NO_EXECUTOR"}
    if executor is not None:
        from core.execution.confidence_gate import GateDecision

        gate_decision = GateDecision(
            action="AUTONOMOUS",  # Human-approved
            confidence=1.0,
            reason="Human operator approved",
            selected_intervention=selected,
            alternatives=[],
        )
        log = await executor.execute(gate_decision, explainer, incident_id)

    # Resolve escalation
    escalator.resolve(incident_id, selected)

    return {
        "incident_id": incident_id,
        "status": "APPROVED_AND_EXECUTED",
        "execution_log": log,
    }
