"""NEXUS — Network state API routes."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/network", tags=["network"])


class TrainUpdate(BaseModel):
    train_id: str
    current_station: str
    next_station: str
    speed_kmph: float = 0
    status: str = "ON_TIME"


@router.get("/state")
async def get_network_state(request: Request):
    """Return all stations, tracks, trains, signals as JSON."""
    driver = request.app.state.neo4j_driver

    with driver.session() as session:
        stations = [
            dict(r["s"])
            for r in session.run("MATCH (s:Station) RETURN s")
        ]
        tracks = [
            {**dict(r["r"]), "from": r["a"]["id"], "to": r["b"]["id"]}
            for r in session.run(
                "MATCH (a:Station)-[r:TRACK]->(b:Station) RETURN a, r, b"
            )
        ]
        trains = [
            dict(r["t"])
            for r in session.run("MATCH (t:Train) RETURN t")
        ]
        signals = [
            dict(r["sig"])
            for r in session.run("MATCH (sig:Signal) RETURN sig")
        ]

    return {
        "stations": stations,
        "tracks": tracks,
        "trains": trains,
        "signals": signals,
    }


@router.get("/trains")
async def get_live_trains(request: Request):
    """Return live train positions with current and next stations."""
    driver = request.app.state.neo4j_driver

    query = """
    MATCH (t:Train)-[:OCCUPIES]->(current:Station)
    OPTIONAL MATCH (t)-[:HEADING_TO]->(next:Station)
    RETURN t.id AS id, t.name AS name, t.status AS status,
           t.speed_kmph AS speed, t.passenger_count AS passengers,
           current.id AS current_station, current.name AS current_station_name,
           next.id AS next_station, next.name AS next_station_name
    """
    with driver.session() as session:
        result = session.run(query)
        trains = [dict(r) for r in result]

    return {"trains": trains}


@router.post("/update-train")
async def update_train(request: Request, update: TrainUpdate):
    """Update a train's position in Neo4j."""
    driver = request.app.state.neo4j_driver

    query = """
    MATCH (t:Train {id: $train_id})
    OPTIONAL MATCH (t)-[old_occ:OCCUPIES]->()
    DELETE old_occ
    WITH t
    OPTIONAL MATCH (t)-[old_head:HEADING_TO]->()
    DELETE old_head
    WITH t
    MATCH (curr:Station {id: $current_station})
    MATCH (next:Station {id: $next_station})
    CREATE (t)-[:OCCUPIES]->(curr)
    CREATE (t)-[:HEADING_TO]->(next)
    SET t.speed_kmph = $speed,
        t.status = $status
    RETURN t.id AS id
    """
    with driver.session() as session:
        result = session.run(
            query,
            train_id=update.train_id,
            current_station=update.current_station,
            next_station=update.next_station,
            speed=update.speed_kmph,
            status=update.status,
        )
        record = result.single()

    return {"updated": record["id"] if record else None}
