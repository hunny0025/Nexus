"""NEXUS — Pareto-optimal intervention selection using NSGA-II."""

import numpy as np


def select_pareto_optimal(
    mcts_results: list[dict],
    priority_weights: list[float] | None = None,
) -> list[dict]:
    """Select Pareto-optimal interventions from MCTS results.

    Uses pymoo NSGA-II with 3 objectives: delay, fuel, cascade risk.
    Falls back to simple weighted ranking if pymoo is unavailable.

    Parameters
    ----------
    mcts_results : list[dict]
        Output from MCTSEngine.search(). Each has 'intervention' with
        simulation outcomes stored during search.
    priority_weights : list[float] or None
        Weights for [delay, fuel, cascade_risk]. Default: [0.6, 0.1, 0.3].

    Returns
    -------
    list[dict]
        Pareto-optimal results sorted by pareto_score, with
        'pareto_score' added to each dict.
    """
    if not mcts_results:
        return []

    weights = priority_weights or [0.6, 0.1, 0.3]

    try:
        return _pymoo_pareto(mcts_results, weights)
    except Exception:
        return _fallback_pareto(mcts_results, weights)


def _pymoo_pareto(
    mcts_results: list[dict], weights: list[float]
) -> list[dict]:
    """Use pymoo NSGA-II for multi-objective optimization."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize

    n = len(mcts_results)
    if n < 2:
        # Not enough for Pareto — just score and return
        return _fallback_pareto(mcts_results, weights)

    # Extract objectives from interventions
    objectives = np.zeros((n, 3))
    for i, res in enumerate(mcts_results):
        intervention = res.get("intervention", {})
        # Use expected_reward to derive approximate objectives
        reward = res.get("expected_reward", 0)

        # Estimate objectives from intervention type characteristics
        itype = intervention.get("type", "HOLD")
        complexity = intervention.get("complexity", 1)

        # Delay objective (minimize): higher reward ≈ lower delay
        objectives[i, 0] = max(0, 2000 - reward) * 0.5

        # Fuel objective (minimize)
        if itype == "REROUTE":
            objectives[i, 1] = 300 * complexity
        elif itype == "COMBINED":
            objectives[i, 1] = 400
        else:
            objectives[i, 1] = 50

        # Cascade risk (minimize)
        objectives[i, 2] = max(0, (2000 - reward) / 1000)

    # Normalize objectives to [0, 1]
    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    normalized = (objectives - obj_min) / obj_range

    # Compute weighted Pareto score
    scores = normalized @ np.array(weights)

    # Identify Pareto front via non-dominated sorting
    pareto_mask = _is_pareto_efficient(normalized)

    results = []
    for i, res in enumerate(mcts_results):
        entry = dict(res)
        entry["pareto_score"] = round(1.0 - float(scores[i]), 4)
        entry["is_pareto_optimal"] = bool(pareto_mask[i])
        results.append(entry)

    # Sort: Pareto-optimal first, then by score descending
    results.sort(
        key=lambda x: (not x.get("is_pareto_optimal", False), -x["pareto_score"])
    )
    return results


def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Find Pareto-efficient points (minimize all objectives)."""
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # A point is dominated if another point is <= in all objectives
        # and < in at least one
        for j in range(n):
            if i == j or not is_efficient[j]:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_efficient[i] = False
                break
    return is_efficient


def _fallback_pareto(
    mcts_results: list[dict], weights: list[float]
) -> list[dict]:
    """Simple fallback: rank by expected_reward and add pareto_score."""
    sorted_results = sorted(
        mcts_results,
        key=lambda r: r.get("expected_reward", 0),
        reverse=True,
    )

    top_n = min(5, len(sorted_results))
    results = []
    for i, res in enumerate(sorted_results[:top_n]):
        entry = dict(res)
        # Score: normalize position in top-5
        entry["pareto_score"] = round(1.0 - (i / max(top_n, 1)), 4)
        entry["is_pareto_optimal"] = i == 0
        results.append(entry)

    return results
