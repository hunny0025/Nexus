"""NEXUS — Async end-to-end demo scenario.

Walks through:
1. System Initialisation & Reset
2. Normal state monitoring
3. Injecting a track fault on a dynamically selected track
4. Polling for anomaly detection, cascade propagation, and escalation brief
5. Simulating operator intervention approval & system stats update
"""

import json
import time
import urllib.request
import urllib.error
import sys

BASE_URL = "http://localhost:8000"

def make_request(url: str, method: str = "GET", data: dict = None) -> dict:
    """Helper to perform HTTP requests to the NEXUS backend."""
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
        json_data = json.dumps(data).encode("utf-8")
        req.data = json_data
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"❌ HTTP Error {e.code}: {body}")
        try:
            return json.loads(body)
        except Exception:
            return {"error": body}
    except Exception as e:
        print(f"❌ Connection error: {e}")
        sys.exit(1)

def main():
    print("=" * 80)
    print(" NEXUS — Railway Intelligence End-to-End Demo Scenario")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PHASE 1: System Initialisation & Reset
    # -------------------------------------------------------------------------
    print("\nPhase 1: Initialising and Resetting Demo State...")
    # Verify health
    health = make_request(f"{BASE_URL}/health")
    if health.get("status") != "healthy":
        print("❌ NEXUS backend is not healthy or not running.")
        sys.exit(1)
    print("✅ NEXUS backend is online and healthy.")

    # Reset
    print("🔄 Triggering demo reset (reseeds database & clears active simulator faults)...")
    reset_res = make_request(f"{BASE_URL}/api/demo/reset", method="POST")
    print(f"✅ Reset outcome: {reset_res.get('status')}")

    # -------------------------------------------------------------------------
    # PHASE 2: Live Network Verification
    # -------------------------------------------------------------------------
    print("\nPhase 2: Inspecting Live Network State...")
    state = make_request(f"{BASE_URL}/api/network/state")
    stations = state.get("stations", [])
    tracks = state.get("tracks", [])
    trains = state.get("trains", [])
    signals = state.get("signals", [])

    print(f"   Stations in Neo4j : {len(stations)}")
    print(f"   Track edges       : {len(tracks)}")
    print(f"   Trains active     : {len(trains)}")
    print(f"   Signals monitored : {len(signals)}")

    if not tracks:
        print("❌ No track sections found in database. Exiting.")
        sys.exit(1)

    # Pick a track to inject a fault on
    target_track = tracks[0]
    track_id = target_track["id"]
    from_stn = target_track["from"]
    to_stn = target_track["to"]
    print(f"👉 Selected target track section: {track_id} ({from_stn} ── {to_stn})")

    # -------------------------------------------------------------------------
    # PHASE 3: Normal Monitoring (Accumulating LSTM Windows)
    # -------------------------------------------------------------------------
    print("\nPhase 3: Monitoring Normal System Operations...")
    print("   (Wait 5 seconds to accumulate sensor readings in sliding windows)...")
    time.sleep(5)
    
    status = make_request(f"{BASE_URL}/api/demo/status")
    print(f"   WebSocket clients active : {status.get('ws_clients_count')}")
    print(f"   Active faults currently  : {len(status.get('active_faults', {}))}")

    # -------------------------------------------------------------------------
    # PHASE 4: Fault Injection & Anomaly Propagation
    # -------------------------------------------------------------------------
    print(f"\nPhase 4: Injecting Sensor Anomalies at {track_id}...")
    injection_data = {
        "location_id": track_id,
        "sensor_types": ["vibration", "track_stress"]
    }
    inj_res = make_request(f"{BASE_URL}/api/demo/inject-fault", method="POST", data=injection_data)
    print(f"⚡ Injection response: {inj_res}")

    print("\n⏳ Waiting for the anomaly to propagate and trigger the confidence gate (approx 12 seconds)...")
    # Loop and poll pending interventions or incidents
    incident_id = None
    for attempt in range(8):
        time.sleep(3)
        print(f"   Polling pending interventions (Attempt {attempt+1}/8)...")
        pending = make_request(f"{BASE_URL}/api/interventions/pending")
        items = pending.get("pending", [])
        if items:
            brief = items[0]
            incident_id = brief["incident_id"]
            print(f"\n🚨 Disruption Incident Flagged: {incident_id}")
            print(f"   Urgency          : {brief['urgency']}")
            print(f"   Cascade Summary  : {brief['cascade_summary']}")
            print(f"   Confidence Score : {brief['confidence']}")
            print(f"   Gate Reason      : {brief['gate_reason']}")
            print(f"   AI Explanation   : {brief['explanation_text']}")
            print("\n💡 Top 3 AI Recommended Interventions:")
            for idx, opt in enumerate(brief["top_3_options"]):
                iv = opt["intervention"]
                proj = opt["projected_outcome"]
                if iv.get("type") == "REROUTE":
                    desc = f"Reroute train {iv.get('train_id')} via {iv.get('selected_route', {}).get('path')}"
                elif iv.get("type") == "HOLD":
                    desc = f"Hold train {iv.get('train_id')} at {iv.get('hold_station')} for {iv.get('estimated_hold_minutes')} mins"
                elif iv.get("type") == "MAINTENANCE_DISPATCH":
                    desc = f"Dispatch crew {iv.get('crew_id')} (ETA: {iv.get('eta_minutes')} mins)"
                else:
                    desc = f"Combined action (synergy bonus: {iv.get('synergy_bonus')})"
                
                print(f"     [{idx}] {iv['type']:25s} | Conf: {opt['confidence']:.2f} | Delay Reduction: {proj['delay_reduction_pct']*100:.0f}% | {desc}")
            break
    
    if not incident_id:
        print("❌ Timed out waiting for anomaly to be confirmed and escalated. Check if MQTT/FastAPI are running.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # PHASE 5: Human Approval & Closed-Loop Learning
    # -------------------------------------------------------------------------
    print(f"\nPhase 5: Simulating Operator Approval of Intervention 0 for {incident_id}...")
    approve_res = make_request(
        f"{BASE_URL}/api/interventions/{incident_id}/approve",
        method="POST",
        data={"intervention_index": 0}
    )
    print(f"✅ Approve response: {approve_res.get('status')}")
    print(f"   Execution Log: {approve_res.get('execution_log', {}).get('status')}")

    print("\n📈 Reviewing Continuous Learning Statistics...")
    stats = make_request(f"{BASE_URL}/api/analytics/system-stats")
    print(f"   Total incidents handled      : {stats.get('total_incidents_handled')}")
    print(f"   Autonomous response rate     : {stats.get('autonomous_rate_pct')}%")
    print(f"   Average AI confidence        : {stats.get('avg_confidence')}")
    print(f"   Average delay reduction rate : {stats.get('avg_delay_reduction_pct')}%")

    print("\n📄 Viewing latest incident in learning log:")
    incidents = make_request(f"{BASE_URL}/api/incidents")
    if incidents.get("incidents"):
        latest = incidents["incidents"][-1]
        print(f"   Incident ID           : {latest['incident_id']}")
        print(f"   Cascade Accuracy      : {latest['cascade_accuracy']*100:.1f}%")
        print(f"   Intervention Accuracy : {latest['intervention_accuracy']*100:.1f}%")
    else:
        print("   No incidents recorded in the learning log history yet.")

    print("\n" + "=" * 80)
    print(" 🎉 NEXUS End-to-End Simulation Demo Completed Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
