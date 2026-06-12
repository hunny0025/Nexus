"""NEXUS Graph Initializer — Seeds Neo4j from JSON data files."""

import json
import os
import random
from pathlib import Path

from neo4j import GraphDatabase

from core.graph.schema import (
    CLEAR_ALL,
    CREATE_ASSIGNED_TO_REL,
    CREATE_CONSTRAINTS,
    CREATE_INDEXES,
    CREATE_MAINTENANCE_CREW,
    CREATE_NEXT_STATION_REL,
    CREATE_OCCUPIES_REL,
    CREATE_SIGNAL,
    CREATE_SIGNAL_AT_REL,
    CREATE_STATION,
    CREATE_TRACK_REL,
    CREATE_TRACK_SECTION,
    CREATE_TRAIN,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
NETWORK_DIR = DATA_DIR / "network"


class GraphInitializer:
    """Reads JSON data files and populates the Neo4j graph database."""

    def __init__(self):
        self.stations = []
        self.tracks = []
        self.trains = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_json(self, path: Path) -> list:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _run(self, tx, query: str, **params):
        tx.run(query, **params)

    # ------------------------------------------------------------------
    # Seeding methods
    # ------------------------------------------------------------------

    def _create_constraints_and_indexes(self, session):
        for q in CREATE_CONSTRAINTS:
            session.run(q)
        for q in CREATE_INDEXES:
            session.run(q)

    def _seed_stations(self, session) -> int:
        self.stations = self._load_json(NETWORK_DIR / "stations.json")
        for s in self.stations:
            session.execute_write(self._run, CREATE_STATION, **s)
        return len(self.stations)

    def _seed_tracks(self, session) -> int:
        self.tracks = self._load_json(NETWORK_DIR / "tracks.json")
        for t in self.tracks:
            session.execute_write(self._run, CREATE_TRACK_REL, **t)
            session.execute_write(self._run, CREATE_TRACK_SECTION, **t)
        return len(self.tracks)

    def _seed_trains(self, session) -> int:
        self.trains = self._load_json(NETWORK_DIR / "trains.json")
        for tr in self.trains:
            session.execute_write(
                self._run,
                CREATE_TRAIN,
                id=tr["id"],
                name=tr["name"],
                speed_kmph=tr["speed_kmph"],
                passenger_count=tr["passenger_count"],
                status=tr["status"],
            )
            # OCCUPIES current station
            session.execute_write(
                self._run,
                CREATE_OCCUPIES_REL,
                train_id=tr["id"],
                station_id=tr["current_station"],
            )
            # HEADING_TO next station
            eta = round(random.uniform(30, 180), 1)
            session.execute_write(
                self._run,
                CREATE_NEXT_STATION_REL,
                train_id=tr["id"],
                next_station_id=tr["next_station"],
                eta_minutes=eta,
            )
        return len(self.trains)

    def _seed_signals(self, session) -> int:
        """Create two signals per station (HOME and STARTER)."""
        count = 0
        for s in self.stations:
            for sig_type in ("HOME", "STARTER"):
                sig_id = f"SIG_{s['id']}_{sig_type}"
                session.execute_write(
                    self._run,
                    CREATE_SIGNAL,
                    id=sig_id,
                    station_id=s["id"],
                    signal_type=sig_type,
                )
                session.execute_write(
                    self._run,
                    CREATE_SIGNAL_AT_REL,
                    signal_id=sig_id,
                    station_id=s["id"],
                )
                count += 1
        return count

    def _seed_maintenance_crews(self, session) -> int:
        """Create 5 maintenance crews at random stations."""
        specializations = [
            "TRACK",
            "SIGNAL",
            "ELECTRICAL",
            "MECHANICAL",
            "GENERAL",
        ]
        crew_stations = random.sample(
            [s["id"] for s in self.stations], k=min(5, len(self.stations))
        )
        for i, station_id in enumerate(crew_stations):
            crew_id = f"CREW_{i + 1:03d}"
            session.execute_write(
                self._run,
                CREATE_MAINTENANCE_CREW,
                id=crew_id,
                name=f"Maintenance Team {i + 1}",
                specialization=specializations[i % len(specializations)],
                current_station_id=station_id,
                response_time_minutes=random.randint(15, 60),
            )
            session.execute_write(
                self._run,
                CREATE_ASSIGNED_TO_REL,
                crew_id=crew_id,
                station_id=station_id,
            )
        return 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed_all(self, driver):
        """Populate the entire Neo4j graph from JSON data files.

        Parameters
        ----------
        driver : neo4j.Driver
            An authenticated Neo4j driver instance.

        Returns
        -------
        dict
            Counts of created entities.
        """
        with driver.session() as session:
            # Wipe existing data
            session.run(CLEAR_ALL)

            # Schema
            self._create_constraints_and_indexes(session)

            # Data
            n_stations = self._seed_stations(session)
            n_tracks = self._seed_tracks(session)
            n_trains = self._seed_trains(session)
            n_signals = self._seed_signals(session)
            n_crews = self._seed_maintenance_crews(session)

        return {
            "stations": n_stations,
            "tracks": n_tracks,
            "trains": n_trains,
            "signals": n_signals,
            "maintenance_crews": n_crews,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    password = os.getenv("NEO4J_PASSWORD", "nexuspassword")

    driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    try:
        initializer = GraphInitializer()
        counts = initializer.seed_all(driver)
        print("✅ Neo4j graph seeded successfully!")
        for entity, count in counts.items():
            print(f"   {entity}: {count}")
    finally:
        driver.close()
