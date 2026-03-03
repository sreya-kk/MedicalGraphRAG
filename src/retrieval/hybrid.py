"""
Hybrid retriever: combines Chroma vector search with Neo4j graph traversal.

Pipeline:
1. Vector search → top-5 chunks
2. Extract entity names from those chunks via Claude mini NER
3. Graph traversal: 1-hop neighbors for each entity
4. Deduplicate + rank merged context
5. Return {chunks, graph_context}
"""

import json
import os
import time
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.graph.neo4j_client import Neo4jClient
from src.retrieval.vector_store import VectorStore

load_dotenv()

NER_MODEL = "claude-haiku-4-5-20251001"
NER_SYSTEM = "You are a named entity recognizer for medical policy text. Return only JSON."
NER_PROMPT = """List the key medical/policy entity names in this text.
Return ONLY a JSON array of short entity name strings (2-6 words each).
Maximum 8 entities. If none found, return [].

Text: {text}"""


def _extract_entities_ner(client: anthropic.Anthropic, texts: list[str]) -> list[str]:
    """
    Quick NER call to Claude to pull entity names from retrieved chunk texts.
    Returns deduplicated list of entity name strings.
    """
    combined = " ".join(texts[:3])[:2000]  # use top-3 chunks, cap chars

    try:
        msg = client.messages.create(
            model=NER_MODEL,
            max_tokens=256,
            system=NER_SYSTEM,
            messages=[{"role": "user", "content": NER_PROMPT.format(text=combined)}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        entities = json.loads(raw)
        if isinstance(entities, list):
            return [str(e) for e in entities if isinstance(e, str) and e.strip()]
        return []
    except Exception:
        return []


def _score_graph_context(
    graph_rows: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    """
    Simple relevance scoring for graph context rows.
    Boosts rows whose source/target appear in the query string.
    """
    query_lower = query.lower()
    scored = []
    for row in graph_rows:
        score = 0.0
        if row.get("source", "").lower() in query_lower:
            score += 1.0
        if row.get("target", "").lower() in query_lower:
            score += 1.0
        scored.append({**row, "_score": score})

    # deduplicate by (source, relation, target)
    seen = set()
    deduped = []
    for row in scored:
        key = (row.get("source"), row.get("relation"), row.get("target"))
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return sorted(deduped, key=lambda x: x["_score"], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        neo4j_client: Neo4jClient | None = None,
        anthropic_client: anthropic.Anthropic | None = None,
        top_k: int = 5,
    ):
        self.vector_store = vector_store or VectorStore()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.anthropic_client = anthropic_client or anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.top_k = top_k
        self._owns_vector = vector_store is None
        self._owns_neo4j = neo4j_client is None

    def connect(self) -> None:
        if self._owns_vector:
            self.vector_store.connect()
        if self._owns_neo4j:
            self.neo4j_client.connect()

    def close(self) -> None:
        if self._owns_neo4j:
            self.neo4j_client.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Full hybrid retrieval.

        Returns:
          {
            "chunks": [...],          # top-k vector results
            "graph_context": [...],   # 1-hop graph rows
            "entities_found": [...]   # entity names used for graph lookup
          }
        """
        # Step 1: Vector search
        chunks = self.vector_store.query(query, n_results=self.top_k)

        # Step 2: NER on retrieved chunks
        chunk_texts = [c["text"] for c in chunks]
        entity_names = _extract_entities_ner(self.anthropic_client, chunk_texts)

        # Step 3: Graph traversal for each entity
        graph_rows: list[dict[str, Any]] = []
        if entity_names:
            # Also try fuzzy DB lookup in case exact names differ
            db_entities = self.neo4j_client.find_entities_by_name(entity_names)
            lookup_names = list({e["name"] for e in db_entities} | set(entity_names))

            for name in lookup_names[:8]:  # cap to avoid N+1 explosion
                neighbors = self.neo4j_client.get_neighbors(name)
                graph_rows.extend(neighbors)

        # Step 4: Score and deduplicate
        graph_context = _score_graph_context(graph_rows, query)[:20]  # top 20

        return {
            "chunks": chunks,
            "graph_context": graph_context,
            "entities_found": entity_names,
        }
