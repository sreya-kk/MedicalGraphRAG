"""
Load extracted entity/relationship data from JSON files into Neo4j.
"""

import json
from pathlib import Path

from src.graph.neo4j_client import Neo4jClient

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_extracted_files() -> list[dict]:
    """Read all *_extracted.json files from data/processed/."""
    files = sorted(PROCESSED_DIR.glob("*_extracted.json"))
    if not files:
        print("No extracted JSON files found. Run extractor first.")
        return []

    all_results = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        print(f"  Loaded {len(data)} chunk results from {f.name}")
        all_results.extend(data)

    return all_results


def build_graph(client: Neo4jClient, results: list[dict]) -> dict:
    """
    Iterate over extraction results and MERGE all entities and relationships
    into Neo4j. Returns counts of operations performed.
    """
    node_count = 0
    rel_count = 0
    skipped_rels = 0

    # First pass: collect all unique entity names per chunk so we can resolve
    # relationship endpoints reliably
    entity_registry: set[str] = set()

    print("  Pass 1: merging entity nodes...")
    for chunk_result in results:
        chunk_id = chunk_result.get("chunk_id", "")
        for entity in chunk_result.get("entities", []):
            name = entity.get("name", "").strip()
            if not name:
                continue
            entity_type = entity.get("type", "Term")
            description = entity.get("description", "")

            client.merge_node(
                name=name,
                entity_type=entity_type,
                description=description,
                chunk_ids=[chunk_id],
            )
            entity_registry.add(name)
            node_count += 1

    print(f"    {node_count} node upserts, {len(entity_registry)} unique entities")

    print("  Pass 2: merging relationships...")
    for chunk_result in results:
        chunk_id = chunk_result.get("chunk_id", "")
        for rel in chunk_result.get("relationships", []):
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            rel_type = rel.get("relation", "REFERENCES")
            evidence = rel.get("evidence", "")

            # Only create relationships between known entities
            if src not in entity_registry or tgt not in entity_registry:
                skipped_rels += 1
                continue

            client.merge_relationship(
                src_name=src,
                rel_type=rel_type,
                tgt_name=tgt,
                evidence=evidence,
                chunk_id=chunk_id,
            )
            rel_count += 1

    print(f"    {rel_count} relationship upserts, {skipped_rels} skipped (unknown endpoints)")

    return {
        "node_upserts": node_count,
        "unique_entities": len(entity_registry),
        "relationship_upserts": rel_count,
        "skipped_relationships": skipped_rels,
    }


def run(neo4j_client: Neo4jClient | None = None) -> dict:
    """Full pipeline: load extracted JSONs → build graph."""
    results = load_extracted_files()
    if not results:
        return {}

    own_client = neo4j_client is None
    if own_client:
        neo4j_client = Neo4jClient()
        neo4j_client.connect()

    try:
        stats = build_graph(neo4j_client, results)
        final_stats = neo4j_client.get_stats()
        print(f"\n  Graph totals: {final_stats['nodes']} nodes, {final_stats['relationships']} relationships")
        return {**stats, **final_stats}
    finally:
        if own_client:
            neo4j_client.close()


if __name__ == "__main__":
    run()
