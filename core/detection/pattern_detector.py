"""NEXUS — Graph-based pattern detector using Neo4j Cypher queries."""

from typing import Optional


class GraphPatternDetector:
    """Detects dangerous precursor patterns in the railway graph.

    Uses three Cypher queries to find:
    a) Degraded tracks with many nearby trains.
    b) Congested stations with delayed trains.
    c) Failing signals at junction stations.

    Parameters
    ----------
    driver : neo4j.Driver
        An authenticated Neo4j driver instance.
    """

    def __init__(self, driver):
        self.driver = driver

    # ------------------------------------------------------------------
    # Query definitions
    # ------------------------------------------------------------------

    QUERY_DEGRADED_TRACK_NEAR_TRAINS = """
    MATCH (s1:Station)-[trk:TRACK]->(s2:Station)
    WHERE trk.health_prob_failed > 0.05
    WITH trk, s1, s2
    MATCH (t:Train)-[:OCCUPIES]->(s)
    WHERE s.id = s1.id OR s.id = s2.id
       OR EXISTS {
         MATCH (s)-[:TRACK*1..2]-(adj:Station)<-[:OCCUPIES]-(t2:Train)
       }
    WITH trk, s1, s2, count(DISTINCT t) AS train_count
    WHERE train_count > 2
    RETURN trk.id AS track_id,
           s1.id AS from_station,
           s2.id AS to_station,
           trk.health_prob_failed AS fail_prob,
           train_count
    """

    QUERY_CONGESTED_STATION_DELAYED = """
    MATCH (t:Train)-[:OCCUPIES]->(s:Station)
    WITH s, collect(t) AS trains, count(t) AS train_count
    WHERE train_count > 3
    WITH s, trains, train_count,
         [tr IN trains WHERE tr.delay_minutes > 0 | tr] AS delayed
    WHERE size(delayed) >= 1
    RETURN s.id AS station_id,
           s.name AS station_name,
           train_count,
           size(delayed) AS delayed_count
    """

    QUERY_FAILING_SIGNAL_AT_JUNCTION = """
    MATCH (sig:Signal)-[:LOCATED_AT]->(s:Station)
    WHERE sig.failure_probability > 0.1 AND s.is_junction = true
    RETURN sig.id AS signal_id,
           s.id AS station_id,
           s.name AS station_name,
           sig.failure_probability AS fail_prob,
           sig.signal_type AS signal_type
    """

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _run_query(self, query: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def detect_precursor_patterns(self) -> list[dict]:
        """Run all three pattern-detection queries.

        Returns
        -------
        list[dict]
            Each dict: {pattern_type, location_id, severity, description}
        """
        patterns = []

        # --- (a) Degraded tracks with nearby trains ---
        try:
            rows = self._run_query(self.QUERY_DEGRADED_TRACK_NEAR_TRAINS)
            for row in rows:
                severity = min(1.0, row["fail_prob"] * 10 + row["train_count"] * 0.1)
                patterns.append(
                    {
                        "pattern_type": "DEGRADED_TRACK_NEAR_TRAINS",
                        "location_id": row["track_id"],
                        "severity": round(severity, 3),
                        "description": (
                            f"Track {row['track_id']} ({row['from_station']}→"
                            f"{row['to_station']}) has fail_prob="
                            f"{row['fail_prob']:.3f} with {row['train_count']} "
                            f"trains within 2 hops"
                        ),
                    }
                )
        except Exception as exc:
            patterns.append(
                {
                    "pattern_type": "DEGRADED_TRACK_NEAR_TRAINS",
                    "location_id": "QUERY_ERROR",
                    "severity": 0.0,
                    "description": f"Query failed: {exc}",
                }
            )

        # --- (b) Congested stations with delayed trains ---
        try:
            rows = self._run_query(self.QUERY_CONGESTED_STATION_DELAYED)
            for row in rows:
                severity = min(
                    1.0, row["train_count"] * 0.15 + row["delayed_count"] * 0.2
                )
                patterns.append(
                    {
                        "pattern_type": "CONGESTED_STATION_WITH_DELAYS",
                        "location_id": row["station_id"],
                        "severity": round(severity, 3),
                        "description": (
                            f"Station {row['station_name']} ({row['station_id']}) has "
                            f"{row['train_count']} trains, {row['delayed_count']} delayed"
                        ),
                    }
                )
        except Exception as exc:
            patterns.append(
                {
                    "pattern_type": "CONGESTED_STATION_WITH_DELAYS",
                    "location_id": "QUERY_ERROR",
                    "severity": 0.0,
                    "description": f"Query failed: {exc}",
                }
            )

        # --- (c) Failing signals at junctions ---
        try:
            rows = self._run_query(self.QUERY_FAILING_SIGNAL_AT_JUNCTION)
            for row in rows:
                severity = min(1.0, row["fail_prob"] * 5)
                patterns.append(
                    {
                        "pattern_type": "FAILING_SIGNAL_AT_JUNCTION",
                        "location_id": row["station_id"],
                        "severity": round(severity, 3),
                        "description": (
                            f"Signal {row['signal_id']} ({row['signal_type']}) at "
                            f"junction {row['station_name']} has failure_prob="
                            f"{row['fail_prob']:.3f}"
                        ),
                    }
                )
        except Exception as exc:
            patterns.append(
                {
                    "pattern_type": "FAILING_SIGNAL_AT_JUNCTION",
                    "location_id": "QUERY_ERROR",
                    "severity": 0.0,
                    "description": f"Query failed: {exc}",
                }
            )

        return patterns
