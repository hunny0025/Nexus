"""NEXUS — Run All Entry Point.

Starts the Uvicorn web server hosting FastAPI, which automatically boots the
sensor simulator, MQTT subscriber, graph database syncer, and orchestrator.
"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure root path is in python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    print("=" * 80)
    print(" NEXUS — Railway Intelligence Platform Runner")
    print("=" * 80)
    print("📡 Starting FastAPI Server on http://localhost:8000")
    print("📡 Press Ctrl+C to stop the platform and all background services.\n")
    
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, log_level="info")
