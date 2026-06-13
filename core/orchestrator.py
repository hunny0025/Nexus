"""NEXUS — LangGraph orchestrator pipeline."""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ── Real ML model singletons (loaded once, used everywhere) ──────────────────
_tft_model    = None
_sage_model   = None
_models_tried = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFT_PATH  = os.path.join(BASE, "data", "models", "tft.pt")
SAGE_PATH = os.path.join(BASE, "data", "models", "graphsage.pt")


def _load_models():
    """Lazy-load trained models (called once on first pipeline invocation)."""
    global _tft_model, _sage_model, _models_tried
    if _models_tried:
        return
    _models_tried = True

    if os.path.exists(TFT_PATH):
        try:
            from core.learning.tft_model import TFTTrainer
            _tft_model = TFTTrainer.load(TFT_PATH)
            logger.info("[OK] TFT model loaded")
        except Exception as e:
            logger.warning(f"TFT load failed: {e}")
    else:
        logger.info("TFT model not found - run scripts/train_models.py first")

    if os.path.exists(SAGE_PATH):
        try:
            from core.learning.graphsage_model import GraphSAGETrainer
            _sage_model = GraphSAGETrainer.load(SAGE_PATH)
            logger.info("[OK] GraphSAGE model loaded")
        except Exception as e:
            logger.warning(f"GraphSAGE load failed: {e}")
    else:
        logger.info("GraphSAGE model not found - using DBN fallback")


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class NexusState(TypedDict, total=False):
    """Full state flowing through the NEXUS pipeline."""

    # Sensor layer
    sensor_readings: dict[str, dict[str, float]]  # location → {sensor: value}
    kalman_results: dict[str, dict]                # sensor_id → validation result

    # Detection layer
    anomaly_confirmed: bool
    anomaly_location: str
    anomaly_score: float
    anomaly_evidence: dict

    # Cascade layer
    cascade_map: dict[str, dict[str, float]]
    high_risk_nodes: list[dict]
    affected_trains: list[str]
    cascade_summary: str
    tft_forecast: dict          # TFT delay forecast for anomaly location
    propagation_source: str     # "graphsage+dbn" | "dbn_only"

    # Counterfactual layer
    intervention_space: list[dict]
    mcts_results: list[dict]
    pareto_results: list[dict]

    # Decision layer
    gate_decision: Any  # GateDecision dataclass
    explanation: str

    # Execution layer
    execution_log: dict
    incident_id: str
    timestamp: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def sensor_fusion(state: NexusState, components: dict) -> NexusState:
    """Fuse sensor readings with Kalman validation."""
    kalman_bank = components["kalman_bank"]
    readings = state.get("sensor_readings", {})

    kalman_results = {}
    for location, sensors in readings.items():
        for sensor_type, value in sensors.items():
            sensor_id = f"{location}_{sensor_type}"
            result = kalman_bank.validate_reading(sensor_id, value)
            kalman_results[sensor_id] = result

    state["kalman_results"] = kalman_results
    return state


def anomaly_detection(state: NexusState, components: dict) -> NexusState:
    """Run LSTM + Kalman fusion to detect anomalies."""
    fusion_engine = components.get("fusion_engine")
    readings = state.get("sensor_readings", {})

    confirmed = False
    best_location = ""
    best_score = 0.0
    best_evidence = {}

    for location, sensors in readings.items():
        sensor_dict = {
            f"{location}_{stype}": val for stype, val in sensors.items()
        }

        # Feed LSTM pipeline
        lstm_pipeline = components.get("lstm_detector")
        if lstm_pipeline is not None:
            vector = [
                sensors.get("vibration", 0.5),
                sensors.get("temperature", 45.0),
                sensors.get("brake_pressure", 6.0),
                sensors.get("wheel_impact", 1.2),
                sensors.get("track_stress", 100.0),
            ]
            lstm_pipeline.add_reading(location, vector)

        if fusion_engine is not None:
            result = fusion_engine.fuse(location, sensor_dict)
            if result["confirmed"] and result["confidence"] > best_score:
                confirmed = True
                best_location = location
                best_score = result["confidence"]
                best_evidence = result["evidence"]
        else:
            # Simplified detection: check Kalman z-scores
            kalman = state.get("kalman_results", {})
            high_z = sum(
                1
                for sid, kr in kalman.items()
                if location in sid and kr.get("z_score", 0) > 2.5
            )
            if high_z >= 2:
                confirmed = True
                best_location = location
                best_score = 0.7
                best_evidence = {"kalman_hits": high_z}

    state["anomaly_confirmed"] = confirmed
    state["anomaly_location"] = best_location
    state["anomaly_score"] = best_score
    state["anomaly_evidence"] = best_evidence
    return state


def tft_forecast(state: NexusState, components: dict) -> NexusState:
    """Run TFT delay forecasting — real ML inference step."""
    _load_models()

    if _tft_model is None or not state.get("anomaly_confirmed"):
        state["tft_forecast"] = {}
        return state

    location = state.get("anomaly_location", "NDLS")
    readings = state.get("sensor_readings", {})

    # Build recent delay history from sensor readings
    from datetime import datetime as dt
    now = dt.now()
    recent_delays = []
    for h in range(24, 0, -1):
        d = readings.get(location, {})
        delay_approx = d.get("temperature", 45.0) * 0.5   # proxy if no delay key
        recent_delays.append({
            "delay_min": d.get("delay_min", delay_approx),
            "hour": (now.hour - h) % 24,
            "day_of_week": now.weekday(),
            "month": now.month,
            "is_junction": 1,
        })

    try:
        forecast = _tft_model.predict(
            station_id=location,
            recent_delays=recent_delays,
        )
        state["tft_forecast"] = forecast
        logger.info(f"TFT forecast for {location}: p50={forecast['p50']}")
    except Exception as e:
        logger.warning(f"TFT inference failed: {e}")
        state["tft_forecast"] = {}

    return state


def cascade_analysis(state: NexusState, components: dict) -> NexusState:
    """Run cascade propagation analysis — GraphSAGE + DBN fallback."""
    _load_models()
    cascade_builder = components.get("cascade_builder")
    if cascade_builder is None or not state.get("anomaly_confirmed"):
        state["cascade_map"] = {}
        state["high_risk_nodes"] = []
        state["affected_trains"] = []
        state["cascade_summary"] = "No anomaly detected"
        return state

    location = state["anomaly_location"]
    score = state.get("anomaly_score", 0.9)

    # ── GraphSAGE real propagation scores ─────────────────────────────────────
    graphsage_scores = {}
    if _sage_model is not None:
        try:
            current_delays = {
                location: score * 180.0,   # convert score to approx minutes
            }
            # Add any known delays from sensor readings
            for sid, sensors in state.get("sensor_readings", {}).items():
                if sid != location:
                    current_delays[sid] = sensors.get("delay_min", 0.0)

            graphsage_scores = _sage_model.predict(current_delays)
            logger.info(f"GraphSAGE scores computed for {len(graphsage_scores)} nodes")
        except Exception as e:
            logger.warning(f"GraphSAGE inference failed: {e}")

    # ── DBN cascade builder (always runs, augmented by GraphSAGE) ─────────────
    cascade_map = cascade_builder.compute_cascade(
        fault_location=location,
        initial_probability=min(1.0, score + 0.1),
    )

    # ── Merge: GraphSAGE overrides DBN where available ────────────────────────
    if graphsage_scores:
        for node_id, gnn_risk in graphsage_scores.items():
            if node_id in cascade_map:
                # Weighted blend: 60% GNN (learned) + 40% DBN (physics)
                dbn_risk = cascade_map[node_id]
                cascade_map[node_id] = round(0.6 * gnn_risk + 0.4 * dbn_risk, 4)
            elif gnn_risk > 0.25:
                cascade_map[node_id] = round(gnn_risk, 4)

        state["propagation_source"] = "graphsage+dbn"
    else:
        state["propagation_source"] = "dbn_only"

    high_risk = cascade_builder.get_high_risk_nodes(cascade_map, threshold=0.3)
    trains = cascade_builder.get_expected_trains_affected(high_risk)
    summary = cascade_builder.summarize(cascade_map)

    # Append TFT forecast info if available
    tft = state.get("tft_forecast", {})
    if tft.get("p50"):
        p50_next = tft["p50"][0] if tft["p50"] else 0
        summary += f" | TFT forecast: +{p50_next:.0f} min delay in 1h (p50)"

    state["cascade_map"] = cascade_map
    state["high_risk_nodes"] = high_risk
    state["affected_trains"] = trains
    state["cascade_summary"] = summary
    return state


def counterfactual(state: NexusState, components: dict) -> NexusState:
    """Generate interventions and run MCTS."""
    generator = components.get("intervention_generator")
    mcts_engine = components.get("mcts_engine")

    if generator is None or not state.get("anomaly_confirmed"):
        state["intervention_space"] = []
        state["mcts_results"] = []
        return state

    location = state["anomaly_location"]
    trains = state.get("affected_trains", [])

    network_state = {
        "total_delay_minutes": len(trains) * 30,
        "cascade_probability": state.get("anomaly_score", 0.5),
        "trains_affected": trains,
        "all_stations": list(state.get("cascade_map", {}).keys()),
    }

    interventions = generator.generate(location, trains, network_state)
    state["intervention_space"] = interventions

    if mcts_engine is not None and interventions:
        mcts_results = mcts_engine.search(network_state, interventions)
        state["mcts_results"] = mcts_results
    else:
        state["mcts_results"] = []

    return state


def pareto(state: NexusState, components: dict) -> NexusState:
    """Apply Pareto optimization to MCTS results."""
    from core.counterfactual.pareto import select_pareto_optimal

    mcts_results = state.get("mcts_results", [])
    if not mcts_results:
        state["pareto_results"] = []
        return state

    pareto_results = select_pareto_optimal(mcts_results)
    state["pareto_results"] = pareto_results
    return state


def decision(state: NexusState, components: dict) -> NexusState:
    """Evaluate confidence gate and generate explanation."""
    gate = components.get("confidence_gate")
    explainer = components.get("explainer")
    learning_agent = components.get("learning_agent")

    pareto_results = state.get("pareto_results", [])
    cascade_map = state.get("cascade_map", {})

    history_score = 0.3
    if learning_agent is not None:
        history_score = learning_agent.get_incident_match_score(
            state.get("cascade_summary", "")
        )

    if gate is not None and pareto_results:
        gate_decision = gate.evaluate(pareto_results, cascade_map, history_score)
    else:
        from core.execution.confidence_gate import GateDecision

        gate_decision = GateDecision(
            action="ESCALATE",
            confidence=0.0,
            reason="No gate or results available",
            selected_intervention={},
            alternatives=[],
        )

    state["gate_decision"] = gate_decision

    # Generate explanation (sync fallback)
    if explainer is not None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    explanation = pool.submit(
                        asyncio.run,
                        explainer.explain(
                            incident_id=state.get("incident_id", "INC_UNKNOWN"),
                            intervention=gate_decision.selected_intervention,
                            gate_decision=gate_decision,
                            cascade_summary=state.get("cascade_summary", ""),
                        ),
                    ).result()
            else:
                explanation = loop.run_until_complete(
                    explainer.explain(
                        incident_id=state.get("incident_id", "INC_UNKNOWN"),
                        intervention=gate_decision.selected_intervention,
                        gate_decision=gate_decision,
                        cascade_summary=state.get("cascade_summary", ""),
                    )
                )
        except Exception:
            explanation = explainer._explain_template(
                state.get("incident_id", "INC_UNKNOWN"),
                gate_decision.selected_intervention,
                gate_decision,
                state.get("cascade_summary", ""),
            )
    else:
        explanation = "No explainer available"

    state["explanation"] = explanation
    return state


async def execution(state: NexusState, components: dict) -> NexusState:
    """Execute or escalate the decision."""
    gate_decision = state.get("gate_decision")
    executor = components.get("executor")
    escalator = components.get("escalator")
    learning_agent = components.get("learning_agent")

    incident_id = state.get("incident_id", f"INC_{uuid.uuid4().hex[:8].upper()}")
    state["incident_id"] = incident_id
    state["timestamp"] = datetime.now(timezone.utc).isoformat()

    if gate_decision is None:
        state["execution_log"] = {"status": "NO_DECISION"}
        return state

    if gate_decision.action == "AUTONOMOUS" and executor is not None:
        explainer = components.get("explainer")
        log = await executor.execute(gate_decision, explainer, incident_id)
        state["execution_log"] = log
    elif gate_decision.action == "ESCALATE" and escalator is not None:
        brief = await escalator.build_brief(
            gate_decision,
            state.get("explanation", ""),
            {
                "cascade_probability": state.get("anomaly_score", 0),
                "trains_affected": state.get("affected_trains", []),
                "cascade_summary": state.get("cascade_summary", ""),
            },
        )
        state["execution_log"] = {
            "status": "ESCALATED",
            "brief": brief,
        }
    else:
        state["execution_log"] = {"status": "NO_EXECUTOR"}

    # Record prediction for learning
    if learning_agent is not None:
        learning_agent.record_prediction(
            incident_id=incident_id,
            predicted_cascade_map=state.get("cascade_map", {}),
            predicted_intervention_outcome=state.get("mcts_results", [{}])[0]
            if state.get("mcts_results")
            else {},
        )

    return state


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_nexus_graph(components: dict):
    """Build the LangGraph pipeline.

    Parameters
    ----------
    components : dict
        Required keys: kalman_bank, lstm_detector, cascade_builder,
        intervention_generator, mcts_engine, confidence_gate,
        explainer, executor, escalator, learning_agent.
        Optional: fusion_engine.

    Returns
    -------
    Compiled LangGraph or a simple callable pipeline as fallback.
    """
    try:
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(NexusState)

        # Add nodes — tft_forecast runs in parallel with anomaly detection result
        workflow.add_node("node_sensor_fusion",    lambda s: sensor_fusion(s, components))
        workflow.add_node("node_anomaly_detection", lambda s: anomaly_detection(s, components))
        workflow.add_node("node_tft_forecast",      lambda s: tft_forecast(s, components))
        workflow.add_node("node_cascade_analysis",  lambda s: cascade_analysis(s, components))
        workflow.add_node("node_counterfactual",    lambda s: counterfactual(s, components))
        workflow.add_node("node_pareto",            lambda s: pareto(s, components))
        workflow.add_node("node_decision",          lambda s: decision(s, components))

        # Edges
        workflow.set_entry_point("node_sensor_fusion")
        workflow.add_edge("node_sensor_fusion", "node_anomaly_detection")

        # Conditional: if anomaly not confirmed -> END
        workflow.add_conditional_edges(
            "node_anomaly_detection",
            lambda s: "node_tft_forecast" if s.get("anomaly_confirmed") else END,
        )
        # TFT forecast feeds into cascade analysis (which uses GNN + DBN)
        workflow.add_edge("node_tft_forecast",     "node_cascade_analysis")
        workflow.add_edge("node_cascade_analysis", "node_counterfactual")
        workflow.add_edge("node_counterfactual",   "node_pareto")
        workflow.add_edge("node_pareto",           "node_decision")
        workflow.add_edge("node_decision",          END)

        compiled = workflow.compile()
        logger.info("LangGraph pipeline compiled: sensor->detect->TFT->GNN+DBN->MCTS->gate->exec")
        return compiled

    except ImportError:
        logger.warning("langgraph not available - using sequential fallback")
        return _build_fallback_pipeline(components)


def _build_fallback_pipeline(components: dict):
    """Simple sequential pipeline when LangGraph is unavailable."""

    class FallbackPipeline:
        def __init__(self, comps):
            self.components = comps

        def invoke(self, state: dict) -> dict:
            state = sensor_fusion(state, self.components)
            state = anomaly_detection(state, self.components)
            if not state.get("anomaly_confirmed"):
                return state
            state = tft_forecast(state, self.components)      # ← real TFT
            state = cascade_analysis(state, self.components)  # ← GraphSAGE+DBN
            state = counterfactual(state, self.components)
            state = pareto(state, self.components)
            state = decision(state, self.components)
            return state

    return FallbackPipeline(components)
