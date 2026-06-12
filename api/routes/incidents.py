"""NEXUS — Incident API routes."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


@router.get("")
async def get_incidents(request: Request):
    """Return last 20 incidents from learning log."""
    learning_agent = getattr(request.app.state, "learning_agent", None)
    if learning_agent is None:
        return {"incidents": []}

    history = learning_agent.get_accuracy_history()
    return {"incidents": history[-20:]}


@router.get("/{incident_id}/cascade")
async def get_cascade(request: Request, incident_id: str):
    """Return cascade map for a specific incident."""
    learning_agent = getattr(request.app.state, "learning_agent", None)
    if learning_agent is None:
        return {"error": "Learning agent not available"}

    record = learning_agent._records.get(incident_id)
    if record is None:
        return {"error": f"Incident {incident_id} not found"}

    return {
        "incident_id": incident_id,
        "cascade_map": record.get("predicted_cascade_map", {}),
    }


@router.get("/{incident_id}/explanation")
async def get_explanation(request: Request, incident_id: str):
    """Return Gemini explanation for an incident."""
    learning_agent = getattr(request.app.state, "learning_agent", None)
    explainer = getattr(request.app.state, "explainer", None)

    if learning_agent is None:
        return {"error": "Learning agent not available"}

    record = learning_agent._records.get(incident_id)
    if record is None:
        return {"error": f"Incident {incident_id} not found"}

    # Try to generate explanation on the fly
    explanation = ""
    if explainer is not None:
        try:
            intervention = record.get("predicted_intervention_outcome", {})
            explanation = await explainer.explain(
                incident_id=incident_id,
                intervention=intervention,
                cascade_summary=f"Incident {incident_id}",
            )
        except Exception as exc:
            explanation = f"Explanation generation failed: {exc}"

    return {
        "incident_id": incident_id,
        "explanation": explanation,
    }
