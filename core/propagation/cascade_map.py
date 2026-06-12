"""NEXUS — Cascade map builder and risk assessor."""

from typing import Optional


class CascadeMapBuilder:
    """Builds cascade maps from belief propagation results and enriches
    them with Neo4j context (nearby trains, etc.).

    Parameters
    ----------
    belief_engine : BeliefPropagationEngine
        The primary networkx-based propagation engine.
    neo4j_driver : neo4j.Driver or None
        Neo4j driver for querying train positions. Optional.
    """

    def __init__(self, belief_engine, neo4j_driver=None):
        self.belief_engine = belief_engine
        self.driver = neo4j_driver

    def compute_cascade(
        self,
        fault_location: str,
        initial_probability: float = 0.9,
        time_steps: int = 4,
    ) -> dict[str, dict[str, float]]:
        """Run propagation and return the full cascade map.

        Returns
        -------
        dict
            {node_id: {"15min": p, "30min": p, "45min": p, "60min": p}}
        """
        return self.belief_engine.propagate(
            fault_node=fault_location,
            fault_probability=initial_probability,
            time_steps=time_steps,
        )

    def get_high_risk_nodes(
        self,
        cascade_map: dict[str, dict[str, float]],
        threshold: float = 0.3,
    ) -> list[dict]:
        """Identify nodes exceeding the risk threshold.

        Returns
        -------
        list[dict]
            Sorted by max_failure_probability descending. Each dict:
            {node_id, max_failure_probability, time_to_peak_risk}
        """
        high_risk = []
        for node_id, time_probs in cascade_map.items():
            max_prob = 0.0
            peak_time = "15min"
            for label, prob in time_probs.items():
                if prob > max_prob:
                    max_prob = prob
                    peak_time = label
            if max_prob >= threshold:
                high_risk.append(
                    {
                        "node_id": node_id,
                        "max_failure_probability": round(max_prob, 4),
                        "time_to_peak_risk": peak_time,
                    }
                )
        high_risk.sort(key=lambda x: x["max_failure_probability"], reverse=True)
        return high_risk

    def get_expected_trains_affected(
        self,
        high_risk_nodes: list[dict],
        neo4j_driver=None,
    ) -> list[str]:
        """Query Neo4j for trains near high-risk nodes.

        Returns list of train IDs.
        """
        driver = neo4j_driver or self.driver
        if driver is None:
            return []

        node_ids = [n["node_id"] for n in high_risk_nodes]
        if not node_ids:
            return []

        query = """
        UNWIND $node_ids AS nid
        OPTIONAL MATCH (t:Train)-[:OCCUPIES]->(s:Station {id: nid})
        WITH collect(DISTINCT t.id) AS at_station
        UNWIND $node_ids AS nid2
        OPTIONAL MATCH (t2:Train)-[:HEADING_TO]->(s2:Station {id: nid2})
        WITH at_station, collect(DISTINCT t2.id) AS heading_to
        WITH at_station + heading_to AS all_trains
        UNWIND all_trains AS tid
        WITH DISTINCT tid WHERE tid IS NOT NULL
        RETURN collect(tid) AS train_ids
        """
        try:
            with driver.session() as session:
                result = session.run(query, node_ids=node_ids)
                record = result.single()
                return record["train_ids"] if record else []
        except Exception:
            return []

    def summarize(self, cascade_map: dict[str, dict[str, float]]) -> str:
        """Return a human-readable one-line summary of the cascade.

        Example
        -------
        "Signal failure at HNZ -> 73% cascade probability to 12 nodes within 60 minutes"
        """
        if not cascade_map:
            return "No cascade data available."

        # Find the origin (highest probability node at 15min)
        origin = max(
            cascade_map.items(),
            key=lambda x: x[1].get("15min", 0.0),
        )
        origin_id = origin[0]

        # Count affected nodes (any probability > 5%)
        affected = 0
        max_prob = 0.0
        for node_id, time_probs in cascade_map.items():
            node_max = max(time_probs.values())
            if node_max > 0.05 and node_id != origin_id:
                affected += 1
            max_prob = max(max_prob, node_max)

        # Find time horizon
        time_labels = list(next(iter(cascade_map.values())).keys())
        horizon = time_labels[-1] if time_labels else "60min"

        return (
            f"Fault at {origin_id} -> {max_prob * 100:.0f}% cascade "
            f"probability to {affected} nodes within {horizon}"
        )
