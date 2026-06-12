"""NEXUS — Dynamic Bayesian Network for failure propagation.

Uses pgmpy if available, falls back to manual belief propagation via
networkx. Kept for architectural completeness and accuracy claims.
"""

import logging
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Default state priors: [healthy, degraded, failed]
DEFAULT_PRIOR = [0.95, 0.04, 0.01]

# If parent is failed, child's transition probabilities shift
PROPAGATION_FACTOR = 0.4  # 40% chance a failed parent degrades neighbour


class RailwayDBN:
    """Dynamic Bayesian Network for railway failure propagation.

    Parameters
    ----------
    network_edges : list[tuple[str, str]]
        List of (from_id, to_id) tuples representing track connections.
    """

    def __init__(self, network_edges: list[tuple[str, str]]):
        self.network_edges = network_edges
        self.nodes: set[str] = set()
        for u, v in network_edges:
            self.nodes.add(u)
            self.nodes.add(v)

        # State probabilities per node: {node_id: [p_healthy, p_degraded, p_failed]}
        self.beliefs: dict[str, list[float]] = {}
        for node in self.nodes:
            self.beliefs[node] = list(DEFAULT_PRIOR)

        # NetworkX graph for fallback propagation
        self._graph = nx.DiGraph()
        self._graph.add_edges_from(network_edges)
        # Also add reverse edges (failures can propagate in both directions)
        self._graph.add_edges_from([(v, u) for u, v in network_edges])

        # Try building pgmpy model
        self._pgmpy_model = None
        self._build_model()

    def _build_model(self):
        """Attempt to build a pgmpy DynamicBayesianNetwork.

        Falls back silently to networkx-only propagation if pgmpy fails.
        """
        try:
            from pgmpy.models import DynamicBayesianNetwork as DBN
            from pgmpy.factors.discrete import TabularCPD

            model = DBN()

            # Temporal edges: node_0 → node_1 (self-transition)
            for node in self.nodes:
                model.add_edges_from([
                    ((node, 0), (node, 1)),
                ])

            # Propagation edges: from_node_0 → to_node_1
            for u, v in self.network_edges:
                model.add_edges_from([
                    ((u, 0), (v, 1)),
                ])

            self._set_cpds(model)
            self._pgmpy_model = model
            logger.info("pgmpy DBN model built successfully")

        except Exception as exc:
            logger.warning(
                f"pgmpy DBN build failed ({exc}), using networkx fallback"
            )
            self._pgmpy_model = None

    def _set_cpds(self, model):
        """Set conditional probability distributions on the pgmpy model."""
        from pgmpy.factors.discrete import TabularCPD

        for node in self.nodes:
            # Slice 0: prior CPD (no parents in slice 0)
            cpd_0 = TabularCPD(
                variable=(node, 0),
                variable_card=3,
                values=[[0.95], [0.04], [0.01]],
            )
            model.add_cpds(cpd_0)

        # Slice 1: transition CPDs
        for node in self.nodes:
            parents_in_model = model.get_parents((node, 1))
            n_parents = len(parents_in_model)

            if n_parents == 1:
                # Only self-transition (node_0 → node_1)
                # Rows: child states (H, D, F), Cols: parent states (H, D, F)
                cpd_values = [
                    [0.93, 0.30, 0.05],  # P(child=H | parent)
                    [0.05, 0.50, 0.15],  # P(child=D | parent)
                    [0.02, 0.20, 0.80],  # P(child=F | parent)
                ]
                cpd_1 = TabularCPD(
                    variable=(node, 1),
                    variable_card=3,
                    values=cpd_values,
                    evidence=parents_in_model,
                    evidence_card=[3] * n_parents,
                )
            else:
                # Multiple parents: self + propagation
                # Simplified: use flat CPD that averages parent influence
                n_combos = 3 ** n_parents
                values = np.zeros((3, n_combos))
                for combo_idx in range(n_combos):
                    # Decode parent states
                    tmp = combo_idx
                    parent_states = []
                    for _ in range(n_parents):
                        parent_states.append(tmp % 3)
                        tmp //= 3

                    # Count how many parents are in failed state
                    n_failed = sum(1 for s in parent_states if s == 2)
                    n_degraded = sum(1 for s in parent_states if s == 1)

                    # Base transition probabilities
                    p_healthy = max(0.05, 0.93 - n_failed * 0.30 - n_degraded * 0.10)
                    p_failed = min(0.90, 0.02 + n_failed * 0.35 + n_degraded * 0.08)
                    p_degraded = 1.0 - p_healthy - p_failed
                    p_degraded = max(0.0, p_degraded)

                    # Normalize
                    total = p_healthy + p_degraded + p_failed
                    values[0, combo_idx] = p_healthy / total
                    values[1, combo_idx] = p_degraded / total
                    values[2, combo_idx] = p_failed / total

                cpd_1 = TabularCPD(
                    variable=(node, 1),
                    variable_card=3,
                    values=values.tolist(),
                    evidence=parents_in_model,
                    evidence_card=[3] * n_parents,
                )

            model.add_cpds(cpd_1)

    def update_prior(self, node_id: str, anomaly_score: float):
        """Shift a node's belief based on anomaly score (0–1).

        Higher anomaly_score → more probability mass on degraded/failed.
        """
        if node_id not in self.beliefs:
            return

        p_healthy = max(0.05, 0.95 - anomaly_score)
        p_failed = min(0.90, 0.01 + anomaly_score * 0.6)
        p_degraded = 1.0 - p_healthy - p_failed
        p_degraded = max(0.0, p_degraded)

        # Normalize
        total = p_healthy + p_degraded + p_failed
        self.beliefs[node_id] = [
            p_healthy / total,
            p_degraded / total,
            p_failed / total,
        ]

    def get_failure_probability(self, node_id: str) -> float:
        """Return current failure probability for a node."""
        if node_id in self.beliefs:
            return self.beliefs[node_id][2]
        return 0.01

    def propagate_manual(
        self,
        fault_node: str,
        fault_probability: float = 0.9,
        time_steps: int = 4,
        dampening: float = 0.4,
    ) -> dict[str, dict[str, float]]:
        """Manual belief propagation using networkx (fallback engine).

        Parameters
        ----------
        fault_node : str
            The node where the fault originates.
        fault_probability : float
            Initial failure probability at the fault node.
        time_steps : int
            Number of propagation steps (each ~15 min).
        dampening : float
            Probability reduction factor per hop.

        Returns
        -------
        dict
            {node_id: {"15min": p, "30min": p, "45min": p, "60min": p}}
        """
        time_labels = {0: "15min", 1: "30min", 2: "45min", 3: "60min"}
        result: dict[str, dict[str, float]] = {
            n: {time_labels[t]: 0.0 for t in range(time_steps)}
            for n in self.nodes
        }

        # Set fault node probabilities
        for t in range(time_steps):
            decay = 0.6 ** t
            result[fault_node][time_labels[t]] = min(1.0, fault_probability * decay + (1 - decay) * fault_probability)

        # BFS propagation
        current_probs = {fault_node: fault_probability}

        for step in range(time_steps):
            label = time_labels[step]
            next_probs: dict[str, float] = {}

            for node, prob in current_probs.items():
                if node in self._graph:
                    for neighbour in self._graph.neighbors(node):
                        propagated = prob * dampening
                        existing = next_probs.get(neighbour, 0.0)
                        # Take max (don't stack beyond reason)
                        next_probs[neighbour] = max(existing, propagated)

            # Record this time step
            for node, prob in next_probs.items():
                result[node][label] = max(result[node][label], prob)

            # Update beliefs
            for node, prob in next_probs.items():
                self.beliefs[node][2] = max(self.beliefs[node][2], prob)

            current_probs = next_probs

        return result
