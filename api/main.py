"""NEXUS Railway Intelligence Platform — API Entry Point."""

import os
import json
import logging
from pathlib import Path
import asyncio
import threading
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from neo4j import GraphDatabase
from api.websocket import WebSocketManager
from api.routes import network, incidents, interventions, analytics

# NEXUS Core Imports
from core.sensors.kalman import KalmanFilterBank
from core.sensors.simulator import SensorSimulator
from core.sensors.mqtt_broker import MQTTSubscriber
from core.sensors.kalman import KalmanFilterBank
from core.detection.lstm_detector import LSTMAnomalyDetector, AnomalyDetectionPipeline
from core.detection.pattern_detector import GraphPatternDetector
from core.detection.fusion import AnomalyFusionEngine
from core.propagation.belief_prop import BeliefPropagationEngine
from core.propagation.cascade_map import CascadeMapBuilder
from core.counterfactual.intervention_space import InterventionSpaceGenerator
from core.counterfactual.simulator import InterventionSimulator
from core.counterfactual.mcts import MCTSEngine
from core.execution.confidence_gate import ConfidenceGate
from core.explainability.gemini_explainer import GeminiExplainer
from core.execution.executor import AutonomousExecutor
from core.execution.escalation import EscalationHandler
from core.learning.post_incident import LearningAgent
from core.orchestrator import build_nexus_graph
from core.graph.initializer import GraphInitializer
from core.graph.sync_agent import GraphSyncAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NEXUS Backend",
    description="AI-powered railway intelligence platform for predictive disruption management.",
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route routers
app.include_router(network.router)
app.include_router(incidents.router)
app.include_router(interventions.router)
app.include_router(analytics.router)

# WebSocket connection manager
ws_manager = WebSocketManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle live WebSocket notifications stream."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"type": "PONG", "payload": data}))
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        await ws_manager.disconnect(websocket)

# Demo Models
class FaultInjectionRequest(BaseModel):
    location_id: str
    sensor_types: List[str]

@app.post("/api/demo/inject-fault")
async def inject_fault(request: Request, body: FaultInjectionRequest):
    """Inject a sensor fault at a specific location."""
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is None:
        raise HTTPException(status_code=400, detail="Simulator is not running or not registered.")
    
    simulator.inject_fault(body.location_id, body.sensor_types)
    
    # Broadcast event
    await ws_manager.broadcast({
        "type": "FAULT_INJECTED",
        "location_id": body.location_id,
        "sensors": body.sensor_types,
        "message": f"Fault injected at {body.location_id} for {body.sensor_types}"
    })
    
    return {"status": "FAULT_INJECTED", "location_id": body.location_id, "sensors": body.sensor_types}

@app.post("/api/demo/reset")
async def reset_demo(request: Request):
    """Reset the demo state: clear faults, reset LSTM window, and reseed Neo4j."""
    # 1. Reset simulator active faults
    simulator = getattr(request.app.state, "simulator", None)
    if simulator is not None:
        simulator._active_faults.clear()
        
    # 2. Reset LSTM pipeline
    lstm_pipeline = getattr(request.app.state, "lstm_detector", None)
    if lstm_pipeline is not None:
        lstm_pipeline.reset()
        
    # 3. Reseed Neo4j
    driver = getattr(request.app.state, "neo4j_driver", None)
    if driver is not None:
        try:
            initializer = GraphInitializer()
            counts = initializer.seed_all(driver)
            logger.info(f"Database re-seeded successfully: {counts}")
        except Exception as e:
            logger.error(f"Failed to reseed database during reset: {e}")
            raise HTTPException(status_code=500, detail=f"Database reset failed: {e}")
            
    # Broadcast reset
    await ws_manager.broadcast({
        "type": "DEMO_RESET",
        "message": "Demo status reset and Neo4j database successfully re-seeded."
    })
    
    return {"status": "RESET_COMPLETED"}

@app.get("/api/demo/status")
async def demo_status(request: Request):
    """Return simulator status, client count, and active faults."""
    simulator = getattr(request.app.state, "simulator", None)
    active_faults = {}
    if simulator is not None:
        active_faults = simulator._active_faults
        
    return {
        "simulator_active": simulator is not None,
        "ws_clients_count": ws_manager.client_count,
        "active_faults": active_faults,
    }

@app.get("/")
async def root():
    return {"service": "NEXUS", "status": "operational", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

async def orchestrator_loop(app_instance):
    """Periodically fetches subscriber readings and invokes the orchestrator."""
    logger.info("Orchestrator loop background task started.")
    await asyncio.sleep(5)  # Wait for sensors to collect initial data
    
    while True:
        try:
            subscriber = getattr(app_instance.state, "subscriber", None)
            orchestrator = getattr(app_instance.state, "orchestrator", None)
            components = getattr(app_instance.state, "components", None)
            
            if subscriber is not None and orchestrator is not None and components is not None:
                sensor_ids = subscriber.get_all_sensor_ids()
                readings = {}
                for sid in sensor_ids:
                    recent = subscriber.get_recent(sid, n=1)
                    if recent:
                        parts = sid.rsplit("_", 1)
                        if len(parts) == 2:
                            loc, stype = parts
                            if loc not in readings:
                                readings[loc] = {}
                            readings[loc][stype] = recent[0]
                
                if readings:
                    state = {"sensor_readings": readings}
                    # Run orchestrator in executor pool to prevent blocking event loop
                    res = await asyncio.to_thread(orchestrator.invoke, state)
                    
                    if res.get("anomaly_confirmed"):
                        # Run the execution node logic (which is async and executes intervention or escalates)
                        from core.orchestrator import execution
                        updated_state = await execution(res, components)
                        
                        # Extract and format gate decision
                        gd = updated_state.get("gate_decision")
                        gd_dict = {}
                        if gd:
                            gd_dict = {
                                "action": gd.action,
                                "confidence": gd.confidence,
                                "reason": gd.reason,
                                "selected_intervention": gd.selected_intervention,
                            }
                            
                        # Broadcast notification to all WS clients
                        await ws_manager.broadcast({
                            "type": "ORCHESTRATOR_ALERT",
                            "anomaly_confirmed": True,
                            "anomaly_location": updated_state.get("anomaly_location"),
                            "anomaly_score": updated_state.get("anomaly_score"),
                            "cascade_summary": updated_state.get("cascade_summary"),
                            "gate_decision": gd_dict,
                            "explanation": updated_state.get("explanation"),
                            "incident_id": updated_state.get("incident_id"),
                        })
        except Exception as e:
            logger.error(f"Error in orchestrator loop: {e}", exc_info=True)
            
        await asyncio.sleep(3)

@app.on_event("startup")
async def startup_event():
    """Initialize all core components and dependency graph on application startup."""
    # 1. Load Environment Variables
    from dotenv import load_dotenv
    load_dotenv()
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "nexuspassword")
    mqtt_broker = os.getenv("MQTT_BROKER", "localhost")
    
    # 2. Initialize Neo4j Driver
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", neo4j_password))
        driver.verify_connectivity()
        app.state.neo4j_driver = driver
        logger.info("Connected to Neo4j database successfully.")
        
        # Check if database is populated, if not, seed it!
        with driver.session() as session:
            count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            if count == 0:
                logger.info("Neo4j database empty. Seeding database...")
                initializer = GraphInitializer()
                initializer.seed_all(driver)
                logger.info("Database seed complete.")
    except Exception as e:
        logger.error(f"Neo4j connection failure: {e}. Running with mock driver fallback.")
        app.state.neo4j_driver = None
        
    # 3. Kalman Filter Bank
    kalman_bank = KalmanFilterBank(anomaly_threshold=3.0)
    app.state.kalman_bank = kalman_bank
    
    # 4. LSTM Anomaly Detector
    model = LSTMAnomalyDetector(input_dim=5, hidden_dim=64, num_layers=2)
    weights_path = Path(__file__).resolve().parent.parent / "data" / "models" / "lstm_weights.pt"
    if weights_path.exists():
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.threshold = checkpoint.get("threshold", 0.1)
            logger.info("Successfully loaded LSTM weights from disk.")
        except Exception as e:
            logger.warning(f"Error loading LSTM weights: {e}. Running with untrained LSTM.")
    else:
        logger.warning("No LSTM weights found. Running with default untrained model.")
        
    lstm_pipeline = AnomalyDetectionPipeline(model)
    app.state.lstm_detector = lstm_pipeline
    
    # 5. Graph Pattern Detector
    pattern_detector = GraphPatternDetector(app.state.neo4j_driver)
    app.state.pattern_detector = pattern_detector
    
    # 6. Anomaly Fusion Engine
    fusion_engine = AnomalyFusionEngine(lstm_pipeline, kalman_bank, pattern_detector)
    app.state.fusion_engine = fusion_engine
    
    # 7. Belief Propagation & Cascade Map Builder
    tracks_path = Path(__file__).resolve().parent.parent / "data" / "network" / "tracks.json"
    edges = []
    if tracks_path.exists():
        try:
            with open(tracks_path, "r", encoding="utf-8") as f:
                tracks = json.load(f)
                edges = [(t["from_id"], t["to_id"]) for t in tracks]
        except Exception as e:
            logger.error(f"Failed to parse tracks.json: {e}")
            
    if not edges:
        logger.warning("Tracks list empty or unavailable. Seeding static fallback connections.")
        edges = [("NDLS", "CNB"), ("HNZ", "CNB"), ("CNB", "ALD")]
        
    belief_engine = BeliefPropagationEngine(edges)
    app.state.belief_engine = belief_engine
    
    cascade_builder = CascadeMapBuilder(belief_engine, app.state.neo4j_driver)
    app.state.cascade_builder = cascade_builder
    
    # 8. Counterfactual intervention components
    intervention_generator = InterventionSpaceGenerator(app.state.neo4j_driver)
    app.state.intervention_generator = intervention_generator
    
    simulator_cf = InterventionSimulator()
    mcts_engine = MCTSEngine(simulator_cf, n_simulations=50)
    app.state.mcts_engine = mcts_engine
    
    # 9. Gate, Explainer, Executor, Escalator, and Learning
    confidence_gate = ConfidenceGate(autonomous_threshold=0.75)
    app.state.confidence_gate = confidence_gate
    
    explainer = GeminiExplainer()
    app.state.explainer = explainer
    
    executor = AutonomousExecutor(app.state.neo4j_driver, ws_manager.broadcast)
    app.state.executor = executor
    
    escalator = EscalationHandler()
    app.state.escalator = escalator
    
    learning_agent = LearningAgent()
    app.state.learning_agent = learning_agent
    
    # 10. Compile Orchestrator LangGraph
    components = {
        "kalman_bank": kalman_bank,
        "lstm_detector": lstm_pipeline,
        "cascade_builder": cascade_builder,
        "intervention_generator": intervention_generator,
        "mcts_engine": mcts_engine,
        "confidence_gate": confidence_gate,
        "explainer": explainer,
        "executor": executor,
        "escalator": escalator,
        "learning_agent": learning_agent,
        "fusion_engine": fusion_engine,
    }
    app.state.components = components
    app.state.orchestrator = build_nexus_graph(components)
    
    # 11. Start MQTT subscriber to receive live telemetry
    try:
        subscriber = MQTTSubscriber(broker_host=mqtt_broker)
        subscriber.start(blocking=False)
        app.state.subscriber = subscriber
        logger.info(f"MQTT Subscriber connected and listening to broker on {mqtt_broker}.")
    except Exception as e:
        logger.error(f"Failed to start MQTT subscriber: {e}")
        app.state.subscriber = None
        
    # 12. Load Track/Train IDs for simulator
    track_ids = [e[0] for e in edges]
    trains_path = Path(__file__).resolve().parent.parent / "data" / "network" / "trains.json"
    train_ids = []
    if trains_path.exists():
        try:
            with open(trains_path, "r", encoding="utf-8") as f:
                train_ids = [t["id"] for t in json.load(f)]
        except Exception as e:
            logger.error(f"Failed to parse trains.json: {e}")
            
    # 13. Start Sensor Telemetry Simulator in background thread
    try:
        simulator = SensorSimulator(
            broker_host=mqtt_broker,
            track_ids=track_ids,
            train_ids=train_ids,
            interval_sec=2.0
        )
        sim_thread = threading.Thread(target=simulator.start, daemon=True)
        sim_thread.start()
        app.state.simulator = simulator
        logger.info("Sensor telemetry simulator thread started successfully.")
    except Exception as e:
        logger.error(f"Failed to start sensor simulator: {e}")
        app.state.simulator = None
        
    # 14. Start Graph Database Sync Agent
    if app.state.neo4j_driver and app.state.subscriber:
        sync_agent = GraphSyncAgent(app.state.neo4j_driver, app.state.subscriber, kalman_bank)
        app.state.sync_agent = sync_agent
        asyncio.create_task(sync_agent.sync_loop(interval=10))
        logger.info("GraphSyncAgent sync task scheduled.")
        
    # 15. Start Orchestrator Loop background task
    asyncio.create_task(orchestrator_loop(app))
    logger.info("NEXUS Orchestration Pipeline assembled and running.")

@app.on_event("shutdown")
async def shutdown_event():
    """Close Neo4j and shutdown all active simulator and MQTT threads."""
    simulator = getattr(app.state, "simulator", None)
    if simulator is not None:
        simulator.stop()
        logger.info("Stopped simulator.")
        
    subscriber = getattr(app.state, "subscriber", None)
    if subscriber is not None:
        subscriber.stop()
        logger.info("Stopped MQTT subscriber.")
        
    sync_agent = getattr(app.state, "sync_agent", None)
    if sync_agent is not None:
        sync_agent.stop()
        logger.info("Stopped Graph sync agent.")
        
    driver = getattr(app.state, "neo4j_driver", None)
    if driver is not None:
        driver.close()
        logger.info("Closed Neo4j driver connection.")
