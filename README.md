# NEXUS: Autonomous Railway Intelligence Platform

NEXUS is an autonomous, real-time railway intelligence platform designed for predictive disruption management and closed-loop decision support. By fusing real-time sensor streams, graph topology, Dynamic Bayesian Networks, Monte Carlo Tree Search, and LLM reasoning, NEXUS detects track/train degradation, models cascade propagation, generates counterfactual interventions, and executes actions autonomously or escalates with human-in-the-loop explainability.

---

## 🏗️ Architecture & Data Flow

```
                                  [ TELEMETRY SIMULATOR ] (Python Thread)
                                             │
                                     (Publish MQTT)
                                             ▼
                                     [ MQTT BROKER ] (Mosquitto)
                                             │
                                    (Subscribe MQTT)
                                             ▼
                                     [ MQTT SUBSCRIBER ]
                                             │
                     ┌───────────────────────┴──────────────────────┐
                     ▼                                              ▼
            [ KALMAN FILTER BANK ]                        [ GRAPH SYNC AGENT ]
                     │ (z-score validation)                         │ (Updates health status)
                     ▼                                              ▼
        [ LSTM AUTOENCODER PIPELINE ] ───► [ ANOMALY FUSION ] ───► [ NEO4J GRAPH DB ]
                                                 │                  (Topology & State)
                                         (Anomaly Confirmed)        
                                                 ▼                  
                                       [ BELIEF PROPAGATION ] ◄───── (Read active tracks)
                                        (Dynamic Cascade Map)
                                                 │
                                                 ▼
                                     [ INTERVENTION SPACE GEN ]
                                     (Reroute, Hold, Dispatch)
                                                 │
                                                 ▼
                                         [ MCTS SIMULATOR ]
                                       (Simulate outcomes)
                                                 │
                                                 ▼
                                       [ PARETO SELECTOR ]
                                      (Multi-objective opt)
                                                 │
                                                 ▼
                                       [ CONFIDENCE GATE ]
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
             (>= 0.75 Confidence)                                    (< 0.75 Confidence)
                     ▼                                                       ▼
           [ AUTONOMOUS EXECUTOR ]                                  [ ESCALATION BRIEF ]
                     │ (Apply changes to Neo4j)                              │ (Pending human approval)
                     ▼                                                       ▼
           [ WEBSOCKET BROADCAST ] ◄─────── (Human Approved) ─────── [ USER/WEB CLIENT ]
                     │
                     ▼
           [ CONTINUOUS LEARNING ] (Accuracy updates & feedback loop)
```

---

## 🛠️ Tech Stack & Workspace Setup

* **Core**: Python 3.10+, FastAPI (Backend), WebSockets (Real-time telemetry & alerts)
* **Databases**: Neo4j (Graph Network Layer), SQLite/Local JSON (Incidents & continuous learning log)
* **Message Broker**: Eclipse Mosquitto (MQTT Telemetry ingestion)
* **ML/AI Logic**: PyTorch (LSTM AutoEncoder), NetworkX (Probabilistic belief propagation), pgmpy (Dynamic Bayesian Networks), PyMoo/MCTS (Reasoning)
* **Explainability**: Google Gemini API (`gemini-pro`)

---

## ⚡ Quick Start Instructions

Follow these exact steps in order in your terminal:

### 1. Prerequisite Infrastructure
Ensure Docker is installed and running. Start the Neo4j and Mosquitto MQTT broker containers:
```bash
docker-compose up -d
```
* Neo4j Browser is accessible at: `http://localhost:7474` (username: `neo4j`, password: `nexuspassword`)
* MQTT Broker port: `1883`

### 2. Python Setup
Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Set up your environment variables:
```bash
copy .env.example .env
# Edit .env and enter your GEMINI_API_KEY
```

### 3. Initialize & Seed Neo4j Database
Populate the railway topology with 15 stations, 20 tracks, 10 trains, signals, and maintenance crews:
```bash
python scripts/seed_graph.py
```

### 4. Train the LSTM Autoencoder
Train the autoencoder model on normal telemetry and calibrate the threshold (saves to `data/models/lstm_weights.pt`):
```bash
python scripts/train_lstm.py
```

### 5. Run the NEXUS Backend Platform
Start uvicorn which runs the FastAPI endpoints, WebSocket endpoint, MQTT Subscriber, Graph Syncer, and Orchestrator:
```bash
python scripts/run_all.py
```

### 6. Run the End-to-End Demo Scenario
In a separate terminal, trigger the automated end-to-end demo script to test the entire pipeline:
```bash
python scripts/demo_scenario.py
```

---

## 📡 API Endpoints Reference

| Category | HTTP Method | Route | Description |
|---|---|---|---|
| **Network** | `GET` | `/api/network/state` | Returns stations, tracks, trains, and signal positions from Neo4j |
| | `GET` | `/api/network/trains` | Returns current and next stations for all trains |
| | `POST` | `/api/network/update-train` | Updates train positions and speeds in Neo4j |
| **Incidents** | `GET` | `/api/incidents` | Retrieves the latest incidents from the learning log |
| | `GET` | `/api/incidents/{incident_id}/cascade` | Returns predicted cascade map for a specific incident |
| | `GET` | `/api/incidents/{incident_id}/explanation` | Generates or retrieves explanation for a disruption |
| **Interventions** | `GET` | `/api/interventions/pending` | Lists all interventions escalated and waiting for human approval |
| | `POST` | `/api/interventions/{incident_id}/approve` | Approves and executes a selected intervention |
| **Analytics** | `GET` | `/api/analytics/learning-curve` | Returns system prediction accuracy history |
| | `GET` | `/api/analytics/system-stats` | Returns aggregated metrics (autonomous rate, delay reduction, etc.) |
| **WebSocket** | `WS` | `/ws` | Live WebSocket connection for sensor values & disruption alerts |
| **Demo Setup** | `POST` | `/api/demo/inject-fault` | Injects telemetry failure on a track or train section |
| | `POST` | `/api/demo/reset` | Resets all active faults and re-seeds Neo4j database |
| | `GET` | `/api/demo/status` | Returns current active simulator faults and WS clients count |

---

## 🧪 Running Individual Verification Tests

You can run individual system integration tests with the following commands:
* **Sensor & Kalman Filter Integration**:
  ```bash
  python scripts/test_sensors.py
  ```
* **Bayesian Cascade Propagation Test**:
  ```bash
  python scripts/test_propagation.py
  ```
* **Counterfactual MCTS Reasoning Test**:
  ```bash
  python scripts/test_counterfactual.py
  ```
