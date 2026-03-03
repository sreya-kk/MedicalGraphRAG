"""
Orchestration script: runs the full GraphRAG ingestion pipeline.

Steps:
  1. Download CMS PDFs (skips if already present)
  2. Chunk PDFs → data/processed/*_chunks.json
  3. Extract entities via Claude → data/processed/*_extracted.json
  4. Build Neo4j graph
  5. Build Chroma vector index
"""

import sys
import time
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def step_banner(n: int, title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Step {n}: {title}")
    print(f"{'=' * 55}")


def main():
    print("\n🏥  Medical GraphRAG — Index Builder")
    print("=" * 55)

    total_start = time.time()

    # ---------------------------------------------------------------
    # Step 1 — Download PDFs
    # ---------------------------------------------------------------
    step_banner(1, "Download CMS PDFs")
    from scripts.download_pdfs import main as download_main

    download_main()

    # ---------------------------------------------------------------
    # Step 2 — Chunk PDFs
    # ---------------------------------------------------------------
    step_banner(2, "Chunk PDFs")
    from src.ingestion.chunker import process_all_pdfs

    chunk_paths = process_all_pdfs()
    if not chunk_paths:
        print("No chunks produced. Exiting.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 3 — Entity + Relationship Extraction
    # ---------------------------------------------------------------
    step_banner(3, "Extract Entities & Relationships (Claude)")
    from src.ingestion.extractor import process_all_chapters

    extracted_paths = process_all_chapters()
    if not extracted_paths:
        print("No extraction results. Exiting.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 4 — Build Neo4j Graph
    # ---------------------------------------------------------------
    step_banner(4, "Build Neo4j Knowledge Graph")
    from src.graph.neo4j_client import Neo4jClient
    from src.graph.builder import run as build_graph_run

    try:
        with Neo4jClient() as neo4j:
            stats = build_graph_run(neo4j_client=neo4j)
        print(f"  Graph: {stats.get('nodes', '?')} nodes, {stats.get('relationships', '?')} relationships")
    except Exception as e:
        print(f"  WARNING: Neo4j step failed: {e}")
        print("  Ensure Neo4j is running: docker-compose up -d")

    # ---------------------------------------------------------------
    # Step 5 — Build Chroma Vector Index
    # ---------------------------------------------------------------
    step_banner(5, "Build Chroma Vector Index")
    from src.retrieval.vector_store import VectorStore

    try:
        with VectorStore() as vstore:
            added = vstore.index_all_chunks()
        print(f"  {added} chunks added to Chroma index")
    except Exception as e:
        print(f"  WARNING: Chroma step failed: {e}")

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    elapsed = time.time() - total_start
    print(f"\n{'=' * 55}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 55}")
    print("\nNext steps:")
    print("  streamlit run src/app.py")
    print()


if __name__ == "__main__":
    main()
