"""NEXUS — Test script for failure propagation engine."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.propagation.belief_prop import BeliefPropagationEngine
from core.propagation.cascade_map import CascadeMapBuilder
from core.propagation.dbn import RailwayDBN


def main():
    print("=" * 70)
    print("NEXUS — Propagation Engine Test")
    print("=" * 70)

    # 5-node test network: A → B → C → D → E, plus A → C (shortcut)
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("A", "C"),
    ]

    t0 = time.time()

    # --- Test 1: BeliefPropagationEngine (primary) ---
    print("\n1. BeliefPropagationEngine (networkx)")
    print("-" * 50)
    engine = BeliefPropagationEngine(edges)
    cascade = engine.propagate(fault_node="A", fault_probability=0.9, time_steps=4)

    for node in sorted(cascade.keys()):
        probs = cascade[node]
        bar = " | ".join(f"{k}={v:.3f}" for k, v in probs.items())
        print(f"  {node}: {bar}")

    affected = engine.get_affected_nodes(cascade, threshold=0.05)
    print(f"\n  Affected nodes (>5%): {affected}")

    # --- Test 2: CascadeMapBuilder ---
    print("\n2. CascadeMapBuilder")
    print("-" * 50)
    builder = CascadeMapBuilder(engine, neo4j_driver=None)
    cascade_map = builder.compute_cascade("A", initial_probability=0.9)
    high_risk = builder.get_high_risk_nodes(cascade_map, threshold=0.3)

    print("  High-risk nodes (>30%):")
    for node in high_risk:
        print(
            f"    {node['node_id']}: max_prob={node['max_failure_probability']:.3f}, "
            f"peak={node['time_to_peak_risk']}"
        )

    summary = builder.summarize(cascade_map)
    print(f"\n  Summary: {summary}")

    # --- Test 3: RailwayDBN (pgmpy + fallback) ---
    print("\n3. RailwayDBN (pgmpy with networkx fallback)")
    print("-" * 50)
    dbn = RailwayDBN(edges)

    dbn.update_prior("A", anomaly_score=0.85)
    print(f"  After update_prior('A', 0.85):")
    print(f"    A beliefs: H={dbn.beliefs['A'][0]:.3f}, D={dbn.beliefs['A'][1]:.3f}, F={dbn.beliefs['A'][2]:.3f}")

    dbn_cascade = dbn.propagate_manual(fault_node="A", fault_probability=0.9)
    print(f"\n  Manual propagation from A:")
    for node in sorted(dbn_cascade.keys()):
        probs = dbn_cascade[node]
        bar = " | ".join(f"{k}={v:.3f}" for k, v in probs.items())
        print(f"    {node}: {bar}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All tests completed in {elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
