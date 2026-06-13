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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Mount the static directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

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
        
    # 2. Reset LSTM pipeline, Kalman filter bank, and subscriber buffers
    lstm_pipeline = getattr(request.app.state, "lstm_detector", None)
    if lstm_pipeline is not None:
        lstm_pipeline.reset()
        
    kalman_bank = getattr(request.app.state, "kalman_bank", None)
    if kalman_bank is not None:
        kalman_bank.reset()
        
    subscriber = getattr(request.app.state, "subscriber", None)
    if subscriber is not None:
        subscriber.reset()
        
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
    return FileResponse("frontend/index.html")

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
                        loc = None
                        stype = None
                        for suffix in ["vibration", "track_stress", "temperature", "brake_pressure", "wheel_impact"]:
                            if sid.endswith(f"_{suffix}"):
                                loc = sid[:-(len(suffix) + 1)]
                                stype = suffix
                                break
                        
                        if loc and stype:
                            if loc not in readings:
                                readings[loc] = {}
                            readings[loc][stype] = recent[0]
                
                if readings:
                    logger.info(f"Orchestrator loop invoking pipeline with {len(readings)} active locations...")
                    state = {"sensor_readings": readings}
                    # Run orchestrator graph
                    if hasattr(orchestrator, "ainvoke"):
                        res = await orchestrator.ainvoke(state)
                    else:
                        res = await asyncio.to_thread(orchestrator.invoke, state)
                    
                    logger.info(f"Orchestrator result: confirmed={res.get('anomaly_confirmed')}, location={res.get('anomaly_location')}, score={res.get('anomaly_score')}")
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

class MockNeo4jRecord:
    def __init__(self, data_dict):
        self._data = data_dict
    def __getitem__(self, key):
        return self._data.get(key)
    def get(self, key, default=None):
        return self._data.get(key, default)
    def keys(self):
        return self._data.keys()
    def values(self):
        return self._data.values()
    def items(self):
        return self._data.items()
    def __repr__(self):
        return repr(self._data)

class MockNeo4jResult:
    def __init__(self, records):
        self._records = [MockNeo4jRecord(r) for r in records]
    def __iter__(self):
        return iter(self._records)
    def single(self):
        return self._records[0] if self._records else None

class MockNeo4jTransaction:
    def __init__(self, session):
        self.session = session
    def run(self, query, **parameters):
        return self.session.run(query, **parameters)

class MockNeo4jSession:
    def __init__(self, driver):
        self.driver = driver
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def execute_write(self, tx_func, *args, **kwargs):
        tx = MockNeo4jTransaction(self)
        return tx_func(tx, *args, **kwargs)
    def execute_read(self, tx_func, *args, **kwargs):
        tx = MockNeo4jTransaction(self)
        return tx_func(tx, *args, **kwargs)
    def run(self, query, **parameters):
        query_norm = " ".join(query.split()).upper()
        
        # 1. Clear database
        if "DETACH DELETE" in query_norm or "CLEAR" in query_norm:
            self.driver.stations = []
            self.driver.tracks = []
            self.driver.trains = []
            self.driver.signals = []
            self.driver.crews = []
            return MockNeo4jResult([])
            
        # 2. Count nodes (seeding verification)
        elif "RETURN COUNT(N)" in query_norm or "RETURN COUNT(S)" in query_norm:
            total_count = len(self.driver.stations) + len(self.driver.tracks) + len(self.driver.trains)
            return MockNeo4jResult([{"c": total_count}])
            
        # 3. Create/Seed Station
        elif "MERGE (S:STATION" in query_norm:
            station_id = parameters.get("id")
            exists = any(s["id"] == station_id for s in self.driver.stations)
            if not exists:
                self.driver.stations.append(parameters)
            return MockNeo4jResult([])
            
        # 4. Create/Seed Track section
        elif "MERGE (TS:TRACKSECTION" in query_norm:
            track_id = parameters.get("id")
            exists = any(t["id"] == track_id for t in self.driver.tracks)
            if not exists:
                track_data = {**parameters, "from": parameters.get("from_id"), "to": parameters.get("to_id")}
                self.driver.tracks.append(track_data)
            return MockNeo4jResult([])
            
        # 5. Create/Seed Train
        elif "MERGE (T:TRAIN" in query_norm:
            train_id = parameters.get("id")
            exists = any(t["id"] == train_id for t in self.driver.trains)
            if not exists:
                train_data = {
                    "id": parameters.get("id"),
                    "name": parameters.get("name"),
                    "speed_kmph": parameters.get("speed_kmph", 0.0),
                    "passenger_count": parameters.get("passenger_count", 0),
                    "status": parameters.get("status", "ON_TIME"),
                    "current_station": "",
                    "next_station": ""
                }
                self.driver.trains.append(train_data)
            return MockNeo4jResult([])
            
        # 6. Seed Signals & Crews
        elif "MERGE (SIG:SIGNAL" in query_norm:
            sig_id = parameters.get("id")
            exists = any(s["id"] == sig_id for s in self.driver.signals)
            if not exists:
                self.driver.signals.append(parameters)
            return MockNeo4jResult([])
            
        elif "MERGE (C:MAINTENANCECREW" in query_norm:
            crew_id = parameters.get("id")
            exists = any(c["id"] == crew_id for c in self.driver.crews)
            if not exists:
                self.driver.crews.append(parameters)
            return MockNeo4jResult([])
            
        # 7. Occupies / Next Station Seeding relationship updates
        elif "OCCUPIES" in query_norm and ("MERGE" in query_norm or "SET" in query_norm or "CREATE" in query_norm):
            train_id = parameters.get("train_id")
            station_id = parameters.get("station_id") or parameters.get("current_station")
            for t in self.driver.trains:
                if str(t["id"]) == str(train_id):
                    if station_id:
                        t["current_station"] = station_id
                    break
            return MockNeo4jResult([])
            
        elif ("NEXT_STATION" in query_norm or "HEADING_TO" in query_norm) and ("MERGE" in query_norm or "SET" in query_norm or "CREATE" in query_norm):
            train_id = parameters.get("train_id")
            next_station_id = parameters.get("next_station_id") or parameters.get("next_station")
            for t in self.driver.trains:
                if str(t["id"]) == str(train_id):
                    if next_station_id:
                        t["next_station"] = next_station_id
                    # Update status
                    if "status" in parameters:
                        if "SENSOR_STATUS" in query_norm:
                            t["sensor_status"] = parameters["status"]
                        else:
                            t["status"] = parameters["status"]
                    elif "REROUTED" in query_norm:
                        t["status"] = "REROUTED"
                    elif "HELD" in query_norm:
                        t["status"] = "HELD"
                    break
            return MockNeo4jResult([])
            
        # 8. Query all Stations
        elif "MATCH (S:STATION) RETURN S" in query_norm:
            return MockNeo4jResult([{"s": s} for s in self.driver.stations])
            
        # 9. Query all Tracks
        elif "MATCH (A:STATION)-[R:TRACK]->(B:STATION) RETURN A, R, B" in query_norm:
            records = []
            for t in self.driver.tracks:
                records.append({
                    "r": t,
                    "a": {"id": t["from"]},
                    "b": {"id": t["to"]}
                })
            return MockNeo4jResult(records)
            
        # 10. Query all Trains
        elif "MATCH (T:TRAIN) RETURN T" in query_norm:
            return MockNeo4jResult([{"t": t} for t in self.driver.trains])
            
        # 11. Query all Signals
        elif "MATCH (SIG:SIGNAL) RETURN SIG" in query_norm:
            return MockNeo4jResult([{"sig": s} for s in self.driver.signals])
            
        # Shortest alternate route query
        elif "SHORTESTPATH" in query_norm:
            train_id = parameters.get("train_id")
            blocked_node = parameters.get("blocked_node")
            
            # Find start and dest stations for this train
            start = ""
            dest = ""
            for t in self.driver.trains:
                if str(t["id"]) == str(train_id):
                    start = t.get("current_station", "")
                    dest = t.get("next_station", "")
                    break
            
            if not start or not dest:
                # Fallback to general stations from the train or network
                start = start or "NDLS"
                dest = dest or "CNB"
                
            from collections import defaultdict
            # Build undirected graph adjacency list from self.driver.tracks
            adj = defaultdict(list)
            for t in self.driver.tracks:
                adj[t["from"]].append((t["to"], t.get("distance_km", 100)))
                adj[t["to"]].append((t["from"], t.get("distance_km", 100)))
                
            paths = []
            def dfs(curr, target, visited, current_path, current_dist):
                if len(paths) >= 3:
                    return
                if curr == target:
                    paths.append((list(current_path), current_dist))
                    return
                for neighbor, dist in adj[curr]:
                    if neighbor == blocked_node:
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        current_path.append(neighbor)
                        dfs(neighbor, target, visited, current_path, current_dist + dist)
                        current_path.pop()
                        visited.remove(neighbor)
            
            visited = {start}
            dfs(start, dest, visited, [start], 0)
            
            # If no path found (or DFS/BFS failed), return a synthetic fallback path
            if not paths:
                synthetic_path = [start, "ALT_STATION", dest]
                if blocked_node in synthetic_path:
                    synthetic_path = [start, dest]
                paths = [(synthetic_path, 300)]
                
            records = []
            for path, dist in paths:
                records.append({
                    "route_path": path,
                    "total_distance": dist
                })
            return MockNeo4jResult(records)
            
        # 12. Live trains positions query or expected trains affected query
        elif "OCCUPIES" in query_norm and "RETURN" in query_norm:
            if "TRAIN_IDS" in query_norm:
                node_ids = parameters.get("node_ids", [])
                affected_tids = []
                for t in self.driver.trains:
                    curr = t.get("current_station", "")
                    nxt = t.get("next_station", "")
                    if curr in node_ids or nxt in node_ids:
                        affected_tids.append(t["id"])
                return MockNeo4jResult([{"train_ids": affected_tids}])
            else:
                records = []
                for t in self.driver.trains:
                    records.append({
                        "id": t["id"],
                        "name": t["name"],
                        "status": t.get("status", "ON_TIME"),
                        "speed": t.get("speed_kmph", 0.0),
                        "passengers": t.get("passenger_count", 500),
                        "current_station": t.get("current_station", ""),
                        "current_station_name": t.get("current_station", ""),
                        "next_station": t.get("next_station", ""),
                        "next_station_name": t.get("next_station", "")
                    })
                return MockNeo4jResult(records)
            
        # 13. Train updates (both position updates and sync agent updates)
        elif "MATCH (T:TRAIN {ID: $TRAIN_ID})" in query_norm or "UPDATE-TRAIN" in query_norm:
            train_id = parameters.get("train_id")
            for t in self.driver.trains:
                if str(t["id"]) == str(train_id):
                    # Only update fields that are provided in parameters, do NOT set missing parameters to None!
                    if "current_station" in parameters:
                        t["current_station"] = parameters["current_station"]
                    if "next_station" in parameters:
                        t["next_station"] = parameters["next_station"]
                    if "speed" in parameters:
                        t["speed_kmph"] = parameters["speed"]
                    if "speed_kmph" in parameters:
                        t["speed_kmph"] = parameters["speed_kmph"]
                    # Update status
                    if "status" in parameters:
                        if "SENSOR_STATUS" in query_norm:
                            t["sensor_status"] = parameters["status"]
                        else:
                            t["status"] = parameters["status"]
                    elif "REROUTED" in query_norm:
                        t["status"] = "REROUTED"
                    elif "HELD" in query_norm:
                        t["status"] = "HELD"
                    if "brake" in parameters:
                        t["last_brake_pressure"] = parameters["brake"]
                    if "temp" in parameters:
                        t["last_temperature"] = parameters["temp"]
                    break
            return MockNeo4jResult([{"id": train_id}])
            
        # Default fallback
        return MockNeo4jResult([])

class MockNeo4jDriver:
    def __init__(self):
        data_dir = Path(__file__).resolve().parent.parent / "data" / "network"
        self.stations = []
        if (data_dir / "stations.json").exists():
            with open(data_dir / "stations.json", "r", encoding="utf-8") as f:
                self.stations = json.load(f)
        self.tracks = []
        if (data_dir / "tracks.json").exists():
            with open(data_dir / "tracks.json", "r", encoding="utf-8") as f:
                raw_tracks = json.load(f)
                self.tracks = [{**t, "from": t["from_id"], "to": t["to_id"]} for t in raw_tracks]
        self.trains = []
        if (data_dir / "trains.json").exists():
            with open(data_dir / "trains.json", "r", encoding="utf-8") as f:
                self.trains = json.load(f)
        self.signals = []
        self.crews = []
    def verify_connectivity(self):
        pass
    def session(self):
        return MockNeo4jSession(self)
    def close(self):
        pass

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
        app.state.neo4j_driver = MockNeo4jDriver()
        
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
    track_ids = []
    if tracks_path.exists():
        try:
            with open(tracks_path, "r", encoding="utf-8") as f:
                track_ids = [t["id"] for t in json.load(f)]
        except Exception as e:
            logger.error(f"Failed to parse tracks.json: {e}")
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
            interval_sec=2.0,
            local_subscriber=app.state.subscriber
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
