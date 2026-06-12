"""NEXUS — Autonomous executor for dispatching interventions."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AutonomousExecutor:
    """Executes approved interventions: updates Neo4j, broadcasts events.

    Parameters
    ----------
    neo4j_driver : neo4j.Driver
        Neo4j driver for graph updates.
    broadcast_fn : Callable
        Async callable to broadcast WebSocket events.
    """

    def __init__(self, neo4j_driver, broadcast_fn: Callable):
        self.driver = neo4j_driver
        self.broadcast = broadcast_fn

    async def execute(
        self,
        gate_decision,
        explainer=None,
        incident_id: Optional[str] = None,
    ) -> dict:
        """Dispatch the selected intervention.

        Parameters
        ----------
        gate_decision : GateDecision
            Output from ConfidenceGate.evaluate().
        explainer : GeminiExplainer or None
            For generating explanations.
        incident_id : str or None
            Unique incident identifier.

        Returns
        -------
        dict
            Execution log entry.
        """
        incident_id = incident_id or f"INC_{uuid.uuid4().hex[:8].upper()}"
        intervention = gate_decision.selected_intervention
        itype = intervention.get("type", "UNKNOWN")

        log_entry = {
            "incident_id": incident_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": gate_decision.action,
            "intervention_type": itype,
            "confidence": gate_decision.confidence,
            "status": "PENDING",
        }

        try:
            if itype == "REROUTE":
                await self._execute_reroute(intervention, incident_id)
            elif itype == "HOLD":
                await self._execute_hold(intervention, incident_id)
            elif itype == "MAINTENANCE_DISPATCH":
                await self._execute_dispatch(intervention, incident_id)
            elif itype == "COMBINED":
                reroute = intervention.get("reroute", {})
                dispatch = intervention.get("dispatch", {})
                if reroute:
                    await self._execute_reroute(reroute, incident_id)
                if dispatch:
                    await self._execute_dispatch(dispatch, incident_id)
            else:
                logger.warning(f"Unknown intervention type: {itype}")

            log_entry["status"] = "EXECUTED"

            # Generate explanation if explainer available
            explanation = ""
            if explainer is not None:
                try:
                    explanation = await explainer.explain(
                        incident_id=incident_id,
                        intervention=intervention,
                        gate_decision=gate_decision,
                    )
                except Exception as exc:
                    explanation = f"Explanation unavailable: {exc}"
            log_entry["explanation"] = explanation

            # Broadcast execution event
            await self.broadcast(
                {
                    "type": "INTERVENTION_EXECUTED",
                    "incident_id": incident_id,
                    "intervention_type": itype,
                    "confidence": gate_decision.confidence,
                    "status": "EXECUTED",
                    "explanation": explanation[:200],
                }
            )

        except Exception as exc:
            log_entry["status"] = "FAILED"
            log_entry["error"] = str(exc)
            logger.error(f"Execution failed for {incident_id}: {exc}")

        return log_entry

    async def _execute_reroute(self, intervention: dict, incident_id: str):
        """Reroute a train: update HEADING_TO relationship in Neo4j."""
        train_id = intervention.get("train_id", "")
        route = intervention.get("selected_route", {})
        new_path = route.get("path", [])

        if not train_id or not new_path:
            logger.warning(f"Reroute skipped — missing train_id or path")
            return

        next_station = new_path[0] if new_path else None
        if next_station is None:
            return

        query = """
        MATCH (t:Train {id: $train_id})-[old:HEADING_TO]->()
        DELETE old
        WITH t
        MATCH (s:Station {id: $next_station})
        CREATE (t)-[:HEADING_TO {eta_minutes: $eta, rerouted: true, incident_id: $incident_id}]->(s)
        SET t.status = 'REROUTED'
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    train_id=train_id,
                    next_station=next_station,
                    eta=route.get("estimated_time_minutes", 90),
                    incident_id=incident_id,
                )
            logger.info(f"Train {train_id} rerouted to {next_station}")
        except Exception as exc:
            logger.error(f"Neo4j reroute failed: {exc}")

    async def _execute_hold(self, intervention: dict, incident_id: str):
        """Hold a train at its current station."""
        train_id = intervention.get("train_id", "")
        hold_minutes = intervention.get("estimated_hold_minutes", 30)

        if not train_id:
            return

        query = """
        MATCH (t:Train {id: $train_id})
        SET t.status = 'HELD',
            t.hold_minutes = $hold_minutes,
            t.held_incident_id = $incident_id
        """
        try:
            with self.driver.session() as session:
                session.run(
                    query,
                    train_id=train_id,
                    hold_minutes=hold_minutes,
                    incident_id=incident_id,
                )
            logger.info(f"Train {train_id} held for {hold_minutes}min")
        except Exception as exc:
            logger.error(f"Neo4j hold failed: {exc}")

    async def _execute_dispatch(self, intervention: dict, incident_id: str):
        """Dispatch a maintenance crew."""
        crew_id = intervention.get("crew_id", "")

        if not crew_id:
            return

        query = """
        MATCH (c:MaintenanceCrew {id: $crew_id})
        SET c.status = 'DISPATCHED',
            c.dispatched_incident_id = $incident_id,
            c.dispatched_at = datetime()
        """
        try:
            with self.driver.session() as session:
                session.run(query, crew_id=crew_id, incident_id=incident_id)
            logger.info(f"Crew {crew_id} dispatched for {incident_id}")
        except Exception as exc:
            logger.error(f"Neo4j dispatch failed: {exc}")
