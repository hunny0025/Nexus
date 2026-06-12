"""NEXUS — Integration test for counterfactual reasoning pipeline."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.counterfactual.intervention_space import InterventionSpaceGenerator
from core.counterfactual.mcts import MCTSEngine
from core.counterfactual.pareto import select_pareto_optimal
from core.counterfactual.simulator import InterventionSimulator


def main():
    print("=" * 70)
    print("NEXUS — Counterfactual Reasoning Pipeline Test")
    print("=" * 70)

    t0 = time.time()

    # Dummy network state
    network_state = {
        "total_delay_minutes": 120,
        "cascade_probability": 0.75,
        "trains_affected": ["12301", "12001", "12259"],
        "fuel_baseline_kg": 5000,
        "all_stations": ["NDLS", "HNZ", "CNB", "ALD", "MGS", "BPL"],
        "train_stations": {
            "12301": "NDLS",
            "12001": "HNZ",
            "12259": "ALD",
        },
    }
    fault_location = "TRK_NDLS_CNB"
    affected_trains = ["12301", "12001", "12259"]

    # --- Step 1: Generate interventions ---
    print("\n1. Generating intervention space …")
    generator = InterventionSpaceGenerator(neo4j_driver=None)
    interventions = generator.generate(
        fault_location=fault_location,
        affected_trains=affected_trains,
        network_state=network_state,
    )
    print(f"   Generated {len(interventions)} interventions:")
    for i, iv in enumerate(interventions):
        print(f"     [{i + 1}] {iv['type']:25s} complexity={iv['complexity']}")

    # --- Step 2: Run MCTS ---
    print(f"\n2. Running MCTS ({50} simulations) …")
    simulator = InterventionSimulator()
    mcts = MCTSEngine(simulator, n_simulations=50)
    mcts_results = mcts.search(
        current_state=network_state,
        intervention_space=interventions,
    )
    print(f"   MCTS returned {len(mcts_results)} scored interventions")
    for i, r in enumerate(mcts_results[:5]):
        print(
            f"     [{i + 1}] {r['intervention']['type']:25s} "
            f"reward={r['expected_reward']:8.1f}  "
            f"sims={r['simulations_run']:3d}  "
            f"conf={r['confidence']:.3f}"
        )

    # --- Step 3: Pareto optimization ---
    print("\n3. Pareto optimization …")
    pareto_results = select_pareto_optimal(mcts_results)
    print(f"   Pareto returned {len(pareto_results)} results")

    print("\n   Top 3 interventions:")
    print(f"   {'#':>3s}  {'Type':25s}  {'Reward':>8s}  {'Pareto':>7s}  {'Optimal':>7s}")
    print(f"   {'---':>3s}  {'---':25s}  {'---':>8s}  {'---':>7s}  {'---':>7s}")
    for i, r in enumerate(pareto_results[:3]):
        print(
            f"   {i + 1:3d}  {r['intervention']['type']:25s}  "
            f"{r['expected_reward']:8.1f}  "
            f"{r['pareto_score']:7.3f}  "
            f"{'YES' if r.get('is_pareto_optimal') else 'no':>7s}"
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All tests completed in {elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
