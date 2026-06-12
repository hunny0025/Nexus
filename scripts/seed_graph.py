"""NEXUS — Seed the Neo4j graph database from network data files."""

import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Ensure project root is on sys.path so core.* imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.graph.initializer import GraphInitializer


def main():
    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    password = os.getenv("NEO4J_PASSWORD", "nexuspassword")

    print(f"Connecting to Neo4j at {uri} …")
    driver = GraphDatabase.driver(uri, auth=("neo4j", password))

    try:
        # Verify connectivity
        driver.verify_connectivity()
        print("Connected ✓")

        initializer = GraphInitializer()
        counts = initializer.seed_all(driver)

        print("\n✅ Graph seeded successfully!")
        print("-" * 35)
        for entity, count in counts.items():
            print(f"  {entity:20s} → {count}")
        print("-" * 35)
        print(f"  {'TOTAL':20s} → {sum(counts.values())}")
    except Exception as exc:
        print(f"\n❌ Seeding failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
