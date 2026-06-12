"""NEXUS Neo4j Graph Schema — Cypher CREATE query constants."""

# ---------------------------------------------------------------------------
# Constraint & Index Queries
# ---------------------------------------------------------------------------

CREATE_CONSTRAINTS = [
    "CREATE CONSTRAINT station_id IF NOT EXISTS FOR (s:Station) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT train_id IF NOT EXISTS FOR (t:Train) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT signal_id IF NOT EXISTS FOR (sig:Signal) REQUIRE sig.id IS UNIQUE",
    "CREATE CONSTRAINT crew_id IF NOT EXISTS FOR (c:MaintenanceCrew) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT track_id IF NOT EXISTS FOR (ts:TrackSection) REQUIRE ts.id IS UNIQUE",
]

CREATE_INDEXES = [
    "CREATE INDEX station_zone IF NOT EXISTS FOR (s:Station) ON (s.zone)",
    "CREATE INDEX train_status IF NOT EXISTS FOR (t:Train) ON (t.status)",
]

# ---------------------------------------------------------------------------
# Node Queries
# ---------------------------------------------------------------------------

CREATE_STATION = """
MERGE (s:Station {id: $id})
SET s.name = $name,
    s.zone = $zone,
    s.lat = $lat,
    s.lng = $lng,
    s.platform_count = $platform_count,
    s.is_junction = $is_junction,
    s.congestion_level = 0.0,
    s.avg_dwell_time_min = 5.0
"""

CREATE_TRAIN = """
MERGE (t:Train {id: $id})
SET t.name = $name,
    t.speed_kmph = $speed_kmph,
    t.passenger_count = $passenger_count,
    t.status = $status,
    t.delay_minutes = 0,
    t.priority = CASE WHEN t.passenger_count > 800 THEN 'HIGH' ELSE 'NORMAL' END
"""

CREATE_SIGNAL = """
MERGE (sig:Signal {id: $id})
SET sig.station_id = $station_id,
    sig.signal_type = $signal_type,
    sig.state = 'GREEN',
    sig.failure_probability = 0.0,
    sig.last_maintenance = datetime()
"""

CREATE_MAINTENANCE_CREW = """
MERGE (c:MaintenanceCrew {id: $id})
SET c.name = $name,
    c.specialization = $specialization,
    c.status = 'AVAILABLE',
    c.current_station_id = $current_station_id,
    c.response_time_minutes = $response_time_minutes
"""

CREATE_TRACK_SECTION = """
MERGE (ts:TrackSection {id: $id})
SET ts.from_id = $from_id,
    ts.to_id = $to_id,
    ts.distance_km = $distance_km,
    ts.max_speed_kmph = $max_speed_kmph,
    ts.track_type = $track_type,
    ts.last_inspection = datetime(),
    ts.age_years = 10
"""

# ---------------------------------------------------------------------------
# Relationship Queries
# ---------------------------------------------------------------------------

CREATE_TRACK_REL = """
MATCH (a:Station {id: $from_id}), (b:Station {id: $to_id})
MERGE (a)-[r:TRACK {id: $id}]->(b)
SET r.distance_km = $distance_km,
    r.max_speed_kmph = $max_speed_kmph,
    r.track_type = $track_type,
    r.current_speed_limit = $max_speed_kmph,
    r.health_prob_healthy = 0.95,
    r.health_prob_degraded = 0.04,
    r.health_prob_failed = 0.01,
    r.vibration_baseline = 0.5,
    r.stress_baseline = 100.0,
    r.occupancy = 0
"""

CREATE_OCCUPIES_REL = """
MATCH (t:Train {id: $train_id}), (s:Station {id: $station_id})
MERGE (t)-[r:OCCUPIES]->(s)
SET r.since = datetime(),
    r.track_position_km = 0.0
"""

CREATE_NEXT_STATION_REL = """
MATCH (t:Train {id: $train_id}), (s:Station {id: $next_station_id})
MERGE (t)-[r:HEADING_TO]->(s)
SET r.eta_minutes = $eta_minutes
"""

CREATE_ASSIGNED_TO_REL = """
MATCH (c:MaintenanceCrew {id: $crew_id}), (s:Station {id: $station_id})
MERGE (c)-[r:ASSIGNED_TO]->(s)
SET r.since = datetime(),
    r.shift = 'DAY'
"""

CREATE_SIGNAL_AT_REL = """
MATCH (sig:Signal {id: $signal_id}), (s:Station {id: $station_id})
MERGE (sig)-[r:LOCATED_AT]->(s)
"""

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

CLEAR_ALL = "MATCH (n) DETACH DELETE n"

DROP_CONSTRAINTS = [
    "DROP CONSTRAINT station_id IF EXISTS",
    "DROP CONSTRAINT train_id IF EXISTS",
    "DROP CONSTRAINT signal_id IF EXISTS",
    "DROP CONSTRAINT crew_id IF EXISTS",
    "DROP CONSTRAINT track_id IF EXISTS",
]
