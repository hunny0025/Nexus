# NEXUS

AI-powered railway intelligence platform for predictive disruption management and real-time decision support.

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/nexus-railway.git
   cd nexus-railway
   ```

2. **Create your environment file**

   ```bash
   cp .env.example .env
   # Edit .env and set your GEMINI_API_KEY
   ```

3. **Install Python dependencies** (for local development)

   ```bash
   pip install -r requirements.txt
   ```

4. **Start infrastructure with Docker Compose**

   ```bash
   docker compose up -d
   ```

## Running

### With Docker (recommended)

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. Neo4j Browser is at `http://localhost:7474`.

### Without Docker (local development)

```bash
uvicorn api.main:app --reload --port 8000
```

## Architecture

```
nexus/
├── core/
│   ├── graph/           # Railway network graph construction & queries (Neo4j)
│   ├── sensors/         # Real-time sensor data ingestion via MQTT
│   ├── detection/       # Anomaly and disruption detection models
│   ├── propagation/     # Delay propagation & cascade simulation
│   ├── counterfactual/  # What-if scenario analysis
│   ├── execution/       # Action execution & optimization (pymoo)
│   ├── explainability/  # Decision explanations (LangChain + Gemini)
│   └── learning/        # Continuous learning & model retraining
├── api/
│   ├── routes/          # FastAPI route definitions
│   └── main.py          # Application entry point
├── data/
│   ├── network/         # Static railway network topology files
│   ├── historical/      # Historical disruption & schedule data
│   └── models/          # Serialized ML model artifacts
├── scripts/             # Utility & data-loading scripts
└── frontend/            # Web dashboard (separate team)
```

**Key Technologies:** Neo4j · PyTorch · FastAPI · MQTT · LangChain · Google Gemini · NetworkX
