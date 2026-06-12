"""NEXUS — Gemini-powered explainability engine."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are NEXUS, an AI railway disruption management system explaining 
your decisions to non-technical railway control room operators.

Guidelines:
- Use plain language. No jargon, no acronyms without explanation.
- Always cite specific numbers: delay minutes, affected trains, probabilities.
- Structure your explanation as: What happened → What we predict → What we recommend → Why.
- Keep it under 200 words.
- Sound confident but acknowledge uncertainty when confidence < 80%."""

PASSENGER_SYSTEM_PROMPT = """You are writing a brief, calm passenger notification for a 
railway disruption. Keep it under 80 words. Include: what happened (vague), expected 
delay, what the railway is doing about it. Be reassuring. No technical details."""


class GeminiExplainer:
    """Generates human-readable explanations using Google Gemini.

    Falls back to template-based explanations if API key is not set.

    Parameters
    ----------
    api_key : str or None
        Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
    model_name : str
        Gemini model to use (default "gemini-pro").
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self._model = None

        if self.api_key and self.api_key != "your_key_here":
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT,
                )
                logger.info("Gemini explainer initialized with API key")
            except Exception as exc:
                logger.warning(f"Gemini init failed: {exc}, using templates")
                self._model = None
        else:
            logger.info("No Gemini API key — using template explanations")

    async def explain(
        self,
        incident_id: str,
        intervention: dict,
        gate_decision=None,
        cascade_summary: str = "",
        **kwargs,
    ) -> str:
        """Generate an explanation of the decision.

        Parameters
        ----------
        incident_id : str
            Unique incident ID.
        intervention : dict
            The selected intervention.
        gate_decision : GateDecision or None
            The confidence gate decision.
        cascade_summary : str
            One-line cascade summary.

        Returns
        -------
        str
            Human-readable explanation.
        """
        if self._model is not None:
            return await self._explain_gemini(
                incident_id, intervention, gate_decision, cascade_summary
            )
        return self._explain_template(
            incident_id, intervention, gate_decision, cascade_summary
        )

    async def _explain_gemini(
        self, incident_id, intervention, gate_decision, cascade_summary
    ) -> str:
        """Call Gemini API for explanation."""
        confidence = gate_decision.confidence if gate_decision else 0.5
        action = gate_decision.action if gate_decision else "UNKNOWN"
        itype = intervention.get("type", "UNKNOWN")

        prompt = f"""Explain this railway incident decision:

Incident ID: {incident_id}
Cascade: {cascade_summary}
Decision: {action} (confidence: {confidence:.0%})
Intervention: {itype}
Details: {intervention}

Write for a control room operator. Be specific with numbers."""

        try:
            import asyncio

            response = await asyncio.to_thread(
                self._model.generate_content, prompt
            )
            return response.text
        except Exception as exc:
            logger.error(f"Gemini API call failed: {exc}")
            return self._explain_template(
                incident_id, intervention, gate_decision, cascade_summary
            )

    def _explain_template(
        self, incident_id, intervention, gate_decision, cascade_summary
    ) -> str:
        """Fallback template-based explanation."""
        itype = intervention.get("type", "UNKNOWN")
        confidence = gate_decision.confidence if gate_decision else 0.5
        action = gate_decision.action if gate_decision else "UNKNOWN"
        train_id = intervention.get("train_id", "affected trains")

        if itype == "REROUTE":
            route = intervention.get("selected_route", {})
            path = " → ".join(route.get("path", ["alternate route"]))
            return (
                f"[{incident_id}] NEXUS detected a disruption. "
                f"{cascade_summary or 'Cascade risk is elevated'}. "
                f"Decision ({action}, {confidence:.0%} confidence): "
                f"Reroute train {train_id} via {path}. "
                f"Estimated additional travel: {route.get('estimated_time_minutes', 'N/A')} min. "
                f"This avoids the affected section and reduces cascade risk."
            )
        elif itype == "HOLD":
            hold_min = intervention.get("estimated_hold_minutes", 30)
            station = intervention.get("hold_station", "current station")
            return (
                f"[{incident_id}] NEXUS detected a disruption. "
                f"{cascade_summary or 'Cascade risk detected'}. "
                f"Decision ({action}, {confidence:.0%} confidence): "
                f"Hold train {train_id} at {station} for ~{hold_min} minutes "
                f"until the track section is confirmed safe."
            )
        elif itype == "MAINTENANCE_DISPATCH":
            crew_id = intervention.get("crew_id", "nearest crew")
            eta = intervention.get("eta_minutes", 30)
            return (
                f"[{incident_id}] NEXUS detected a disruption. "
                f"{cascade_summary or 'Track degradation detected'}. "
                f"Decision ({action}, {confidence:.0%} confidence): "
                f"Dispatch {crew_id} (ETA: {eta} min) to inspect and repair. "
                f"This addresses the root cause and prevents further cascade."
            )
        elif itype == "COMBINED":
            return (
                f"[{incident_id}] NEXUS detected a serious disruption. "
                f"{cascade_summary or 'High cascade risk'}. "
                f"Decision ({action}, {confidence:.0%} confidence): "
                f"Combined response — rerouting critical trains AND dispatching "
                f"maintenance crew. This dual approach maximizes safety."
            )
        else:
            return (
                f"[{incident_id}] NEXUS responded to a disruption with "
                f"{itype} intervention ({confidence:.0%} confidence)."
            )

    async def generate_passenger_notification(
        self,
        incident_id: str,
        delay_minutes: int,
        affected_route: str = "",
    ) -> str:
        """Generate a passenger-facing notification.

        Returns
        -------
        str
            Brief, reassuring passenger notification.
        """
        if self._model is not None:
            try:
                import asyncio
                import google.generativeai as genai

                passenger_model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=PASSENGER_SYSTEM_PROMPT,
                )
                prompt = (
                    f"Write a passenger notification for: "
                    f"Route {affected_route or 'this service'} is experiencing "
                    f"a delay of approximately {delay_minutes} minutes due to "
                    f"a technical issue. Alternative arrangements are being made."
                )
                response = await asyncio.to_thread(
                    passenger_model.generate_content, prompt
                )
                return response.text
            except Exception:
                pass

        # Template fallback
        return (
            f"Attention passengers: We apologise for the inconvenience. "
            f"{'Services on ' + affected_route + ' are' if affected_route else 'Your service is'} "
            f"currently experiencing a delay of approximately {delay_minutes} minutes "
            f"due to a technical issue. Our team is working to resolve this as "
            f"quickly as possible. We appreciate your patience."
        )
