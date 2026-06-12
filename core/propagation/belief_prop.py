"""NEXUS — NetworkX-based belief propagation engine (primary engine)."""

import networkx as nx


class BeliefPropagationEngine:
    """Propagates failure probabilities through the railway network.

    Uses networkx DiGraph as a reliable, fast propagation engine.

    Parameters
    ----------
    network_edges : list[tuple[str, str]]
        List of (from_id, to_id) tuples representing track connections.
    dampening_per_hop : float
        Probability reduction per hop (default 0.4).
    decay_per_step : float
        Temporal decay per time step (default 0.6).
    """

    def __init__(
        self,
        network_edges: list[tuple[str, str]],
        dampening_per_hop: float = 0.4,
        decay_per_step: float = 0.6,
    ):
        self.dampening_per_hop = dampening_per_hop
        self.decay_per_step = decay_per_step

        self.graph = nx.DiGraph()
        self.graph.add_edges_from(network_edges)
        # Bidirectional — failures propagate both ways
        self.graph.add_edges_from([(v, u) for u, v in network_edges])

        self.nodes = set(self.graph.nodes())

    def propagate(
        self,
        fault_node: str,
        fault_probability: float = 0.9,
        time_steps: int = 4,
    ) -> dict[str, dict[str, float]]:
        """Propagate failure probability from a fault node over time.

        Parameters
        ----------
        fault_node : str
            Origin node of the fault.
        fault_probability : float
            Initial failure probability at origin (default 0.9).
        time_steps : int
            Number of 15-minute time steps to simulate (default 4 = 60min).

        Returns
        -------
        dict
            {node_id: {"15min": p, "30min": p, "45min": p, "60min": p}}
        """
        time_labels = [f"{(i + 1) * 15}min" for i in range(time_steps)]

        # Initialize result map
        result: dict[str, dict[str, float]] = {
            node: {label: 0.0 for label in time_labels}
            for node in self.nodes
        }

        # Fault node retains high probability with temporal decay
        for step in range(time_steps):
            decay = self.decay_per_step ** step
            result[fault_node][time_labels[step]] = round(
                fault_probability * (1.0 - (1.0 - decay) * 0.3), 4
            )

        # BFS propagation at each time step
        # Track cumulative reach: node → max probability at each step
        current_frontier: dict[str, float] = {fault_node: fault_probability}

        for step in range(time_steps):
            label = time_labels[step]
            next_frontier: dict[str, float] = {}

            for node, prob in current_frontier.items():
                if node not in self.graph:
                    continue
                for neighbour in self.graph.neighbors(node):
                    propagated = prob * self.dampening_per_hop
                    if propagated < 0.01:
                        continue  # too small to matter
                    existing = next_frontier.get(neighbour, 0.0)
                    next_frontier[neighbour] = max(existing, propagated)

            # Record probabilities for this time step
            for node, prob in next_frontier.items():
                result[node][label] = max(result[node][label], round(prob, 4))

            # Apply temporal decay and merge into next iteration
            decayed_frontier: dict[str, float] = {}
            for node, prob in next_frontier.items():
                decayed = prob * self.decay_per_step
                if decayed >= 0.01:
                    decayed_frontier[node] = decayed
            # Keep fault node active
            decayed_frontier[fault_node] = (
                current_frontier.get(fault_node, fault_probability)
                * self.decay_per_step
            )
            current_frontier = decayed_frontier

        return result

    def get_affected_nodes(
        self, cascade_map: dict[str, dict[str, float]], threshold: float = 0.05
    ) -> list[str]:
        """Return nodes with any time-step probability above threshold."""
        affected = []
        for node, time_probs in cascade_map.items():
            if any(p >= threshold for p in time_probs.values()):
                affected.append(node)
        return affected
