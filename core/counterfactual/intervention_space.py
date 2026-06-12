"""NEXUS — Intervention space generator for counterfactual reasoning."""

import random
from typing import Optional


class InterventionSpaceGenerator:
    """Generates possible interventions for a given fault scenario.

    Parameters
    ----------
    neo4j_driver : neo4j.Driver or None
        Neo4j driver for route and crew queries.
    """

    def __init__(self, neo4j_driver=None):
        self.driver = neo4j_driver

    def generate(
        self,
        fault_location: str,
        affected_trains: list[str],
        network_state: dict,
    ) -> list[dict]:
        """Generate all viable interventions.

        Parameters
        ----------
        fault_location : str
            The node where the fault occurred.
        affected_trains : list[str]
            Train IDs currently affected.
        network_state : dict
            Current state snapshot (used for context).

        Returns
        -------
        list[dict]
            Each dict has: type, complexity, and type-specific fields.
        """
        interventions = []

        # 1. REROUTE — one per affected train
        for train_id in affected_trains:
            alt_routes = self._get_alternate_routes(
                train_id, fault_location, network_state
            )
            if alt_routes:
                interventions.append(
                    {
                        "type": "REROUTE",
                        "complexity": 2,
                        "train_id": train_id,
                        "alternate_routes": alt_routes,
                        "selected_route": alt_routes[0],
                        "estimated_delay_reduction_pct": random.uniform(0.3, 0.6),
                    }
                )

        # 2. HOLD — one per affected train
        for train_id in affected_trains:
            interventions.append(
                {
                    "type": "HOLD",
                    "complexity": 1,
                    "train_id": train_id,
                    "hold_station": network_state.get("train_stations", {}).get(
                        train_id, "UNKNOWN"
                    ),
                    "estimated_hold_minutes": random.randint(25, 45),
                }
            )

        # 3. MAINTENANCE_DISPATCH — one per available crew
        crews = self._get_available_crews(fault_location, network_state)
        for crew in crews:
            interventions.append(
                {
                    "type": "MAINTENANCE_DISPATCH",
                    "complexity": 2,
                    "crew_id": crew["id"],
                    "crew_station": crew["station"],
                    "eta_minutes": crew["eta"],
                    "specialization": crew.get("specialization", "GENERAL"),
                }
            )

        # 4. COMBINED — top reroute + top dispatch
        reroutes = [i for i in interventions if i["type"] == "REROUTE"]
        dispatches = [i for i in interventions if i["type"] == "MAINTENANCE_DISPATCH"]
        if reroutes and dispatches:
            interventions.append(
                {
                    "type": "COMBINED",
                    "complexity": 3,
                    "reroute": reroutes[0],
                    "dispatch": dispatches[0],
                    "synergy_bonus": 0.1,
                }
            )

        return interventions

    def _get_alternate_routes(
        self,
        train_id: str,
        fault_location: str,
        network_state: dict,
    ) -> list[dict]:
        """Query Neo4j for alternate routes avoiding the blocked node.

        Falls back to synthetic routes if Neo4j unavailable.
        """
        if self.driver is not None:
            try:
                return self._query_neo4j_routes(train_id, fault_location)
            except Exception:
                pass

        # Synthetic fallback: generate 2 plausible routes
        stations = network_state.get("all_stations", ["ALT_A", "ALT_B", "ALT_C"])
        safe_stations = [s for s in stations if s != fault_location][:4]

        routes = []
        for i in range(min(2, len(safe_stations) - 1)):
            route_nodes = safe_stations[i : i + 3] if i + 3 <= len(safe_stations) else safe_stations[i:]
            routes.append(
                {
                    "route_id": f"ALT_{train_id}_{i + 1}",
                    "path": route_nodes,
                    "distance_km": random.randint(200, 600),
                    "estimated_time_minutes": random.randint(60, 240),
                }
            )
        return routes

    def _query_neo4j_routes(
        self, train_id: str, fault_location: str
    ) -> list[dict]:
        """Query Neo4j for shortest alternative paths."""
        query = """
        MATCH (t:Train {id: $train_id})-[:OCCUPIES]->(start:Station)
        MATCH (t)-[:HEADING_TO]->(dest:Station)
        MATCH path = shortestPath(
            (start)-[:TRACK*..10]->(dest)
        )
        WHERE NONE(n IN nodes(path) WHERE n.id = $blocked_node)
        RETURN [n IN nodes(path) | n.id] AS route_path,
               reduce(d = 0, r IN relationships(path) | d + r.distance_km) AS total_distance
        LIMIT 3
        """
        routes = []
        with self.driver.session() as session:
            result = session.run(
                query, train_id=train_id, blocked_node=fault_location
            )
            for i, record in enumerate(result):
                routes.append(
                    {
                        "route_id": f"ALT_{train_id}_{i + 1}",
                        "path": record["route_path"],
                        "distance_km": record["total_distance"] or random.randint(200, 600),
                        "estimated_time_minutes": random.randint(60, 240),
                    }
                )
        return routes

    def _get_available_crews(
        self, fault_location: str, network_state: dict
    ) -> list[dict]:
        """Query Neo4j for available maintenance crews."""
        if self.driver is not None:
            try:
                return self._query_neo4j_crews(fault_location)
            except Exception:
                pass

        # Synthetic fallback
        return [
            {
                "id": f"CREW_{i:03d}",
                "station": f"STN_{i}",
                "eta": random.randint(15, 60),
                "specialization": random.choice(
                    ["TRACK", "SIGNAL", "ELECTRICAL", "MECHANICAL"]
                ),
            }
            for i in range(1, 4)
        ]

    def _query_neo4j_crews(self, fault_location: str) -> list[dict]:
        """Query Neo4j for AVAILABLE maintenance crews."""
        query = """
        MATCH (c:MaintenanceCrew)-[:ASSIGNED_TO]->(s:Station)
        WHERE c.status = 'AVAILABLE'
        RETURN c.id AS id, s.id AS station, c.response_time_minutes AS eta,
               c.specialization AS specialization
        """
        crews = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                crews.append(
                    {
                        "id": record["id"],
                        "station": record["station"],
                        "eta": record["eta"] or random.randint(15, 60),
                        "specialization": record["specialization"] or "GENERAL",
                    }
                )
        return crews
