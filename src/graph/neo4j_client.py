"""
Neo4j connection and CRUD helpers.
Uses MERGE to avoid duplicates; creates indexes on Entity(name) and Entity(type).
"""

import os
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver, Session

load_dotenv()


class Neo4jClient:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self._driver: Driver | None = None

    def connect(self) -> None:
        """Open the driver connection and ensure indexes exist."""
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        self._driver.verify_connectivity()
        self._ensure_indexes()

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def _session(self) -> Session:
        if not self._driver:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._driver.session()

    def _ensure_indexes(self) -> None:
        """Create indexes for fast entity lookups."""
        with self._session() as session:
            session.run(
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )
            session.run(
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)"
            )

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def merge_node(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        chunk_ids: list[str] | None = None,
    ) -> None:
        """
        MERGE an Entity node by name. On creation set type/description.
        On match, append new chunk_ids to the existing list.
        """
        with self._session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET
                    e.type = $type,
                    e.description = $description,
                    e.chunk_ids = $chunk_ids
                ON MATCH SET
                    e.chunk_ids = [x IN e.chunk_ids + $chunk_ids WHERE x IS NOT NULL]
                """,
                name=name,
                type=entity_type,
                description=description,
                chunk_ids=chunk_ids or [],
            )

    def merge_relationship(
        self,
        src_name: str,
        rel_type: str,
        tgt_name: str,
        evidence: str = "",
        chunk_id: str = "",
    ) -> None:
        """
        MERGE a directed relationship between two Entity nodes.
        Both nodes must already exist (call merge_node first).
        rel_type must be a valid Neo4j relationship type (uppercase, no spaces).
        """
        # Sanitize rel_type: uppercase, replace spaces/hyphens with underscores
        safe_rel = rel_type.upper().replace(" ", "_").replace("-", "_")

        with self._session() as session:
            session.run(
                f"""
                MATCH (a:Entity {{name: $src}}), (b:Entity {{name: $tgt}})
                MERGE (a)-[r:{safe_rel}]->(b)
                ON CREATE SET r.evidence = $evidence, r.chunk_ids = [$chunk_id]
                ON MATCH SET r.chunk_ids = r.chunk_ids + [$chunk_id]
                """,
                src=src_name,
                tgt=tgt_name,
                evidence=evidence,
                chunk_id=chunk_id,
            )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_neighbors(self, entity_name: str, hops: int = 1) -> list[dict[str, Any]]:
        """
        Return up to `hops`-hop neighbors of an entity with relationship info.
        Returns list of {source, relation, target, evidence}.
        """
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (a:Entity {{name: $name}})-[r]->(b:Entity)
                RETURN
                    a.name AS source,
                    type(r) AS relation,
                    b.name AS target,
                    r.evidence AS evidence,
                    b.type AS target_type,
                    b.description AS target_description
                LIMIT 50
                """,
                name=entity_name,
            )
            return [dict(record) for record in result]

    def get_node_count(self) -> int:
        with self._session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS cnt")
            return result.single()["cnt"]

    def get_relationship_count(self) -> int:
        with self._session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            return result.single()["cnt"]

    def get_stats(self) -> dict[str, int]:
        return {
            "nodes": self.get_node_count(),
            "relationships": self.get_relationship_count(),
        }

    def find_entities_by_name(self, names: list[str]) -> list[dict[str, Any]]:
        """Fuzzy-ish lookup: match entities whose name contains any of the given strings."""
        with self._session() as session:
            results = []
            for name in names:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    RETURN e.name AS name, e.type AS type, e.description AS description
                    LIMIT 5
                    """,
                    name=name,
                )
                results.extend([dict(r) for r in result])
            return results

    def clear_all(self) -> None:
        """Delete all nodes and relationships. Use with caution."""
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def get_subgraph(
        self,
        limit: int = 100,
        entity_types: list[str] | None = None,
        focus_entity: str | None = None,
    ) -> dict[str, list]:
        """
        Return a subgraph for visualization.

        - focus_entity set: return the neighborhood of that entity
        - entity_types set: filter nodes to those types
        - default: return up to `limit` edges across the whole graph

        Returns {"nodes": [{id, name, type, description}], "edges": [{source, target, relation, evidence}]}
        """
        with self._session() as session:
            if focus_entity:
                if entity_types:
                    result = session.run(
                        """
                        MATCH (a:Entity {name: $name})-[r]-(b:Entity)
                        WHERE a.type IN $types AND b.type IN $types
                        RETURN a, r, b
                        LIMIT $limit
                        """,
                        name=focus_entity,
                        types=entity_types,
                        limit=limit,
                    )
                else:
                    result = session.run(
                        """
                        MATCH (a:Entity {name: $name})-[r]-(b:Entity)
                        RETURN a, r, b
                        LIMIT $limit
                        """,
                        name=focus_entity,
                        limit=limit,
                    )
            elif entity_types:
                result = session.run(
                    """
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE a.type IN $types AND b.type IN $types
                    RETURN a, r, b
                    LIMIT $limit
                    """,
                    types=entity_types,
                    limit=limit,
                )
            else:
                result = session.run(
                    """
                    MATCH (a:Entity)-[r]->(b:Entity)
                    RETURN a, r, b
                    LIMIT $limit
                    """,
                    limit=limit,
                )

            nodes: dict[str, dict] = {}
            edges: list[dict] = []

            for record in result:
                a = record["a"]
                b = record["b"]
                r = record["r"]

                for node in (a, b):
                    nid = node.element_id
                    if nid not in nodes:
                        nodes[nid] = {
                            "id": nid,
                            "name": node.get("name", ""),
                            "type": node.get("type", ""),
                            "description": node.get("description", ""),
                        }

                edges.append(
                    {
                        "source": a.element_id,
                        "target": b.element_id,
                        "relation": r.type,
                        "evidence": r.get("evidence", ""),
                    }
                )

            return {"nodes": list(nodes.values()), "edges": edges}
