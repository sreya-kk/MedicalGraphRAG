import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from src.graph.neo4j_client import Neo4jClient
from src.graph.builder import run as build_graph_run
from src.retrieval.vector_store import VectorStore

print("Building Neo4j graph...")
with Neo4jClient() as neo4j:
    stats = build_graph_run(neo4j_client=neo4j)
    print(stats)

print("\nBuilding Chroma index...")
with VectorStore() as vstore:
    added = vstore.index_all_chunks()
    print(f"{added} chunks indexed")

print("\nDone!")
