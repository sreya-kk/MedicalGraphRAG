"""
Chroma vector store for chunk retrieval.
Embedding model: sentence-transformers/all-MiniLM-L6-v2
"""

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_CHROMA_DIR = str(DATA_DIR / "chroma")
COLLECTION_NAME = "medical_policy_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or os.environ.get(
            "CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR
        )
        self._client: chromadb.PersistentClient | None = None
        self._collection = None
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

    def connect(self) -> None:
        """Open (or create) the persistent Chroma collection."""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        pass  # chromadb PersistentClient auto-flushes

    @property
    def collection(self):
        if self._collection is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._collection

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 100) -> int:
        """
        Add text chunks to the collection.
        Skips chunks whose IDs already exist.
        Returns number of chunks added.
        """
        existing_ids = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            return 0

        added = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            self.collection.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "chapter": c.get("chapter", ""),
                        "page_num": c.get("page_num", 0),
                        "source_url": c.get("source_url", ""),
                    }
                    for c in batch
                ],
            )
            added += len(batch)

        return added

    def index_all_chunks(self) -> int:
        """Load all *_chunks.json files and add to the collection."""
        chunk_files = sorted(PROCESSED_DIR.glob("*_chunks.json"))
        if not chunk_files:
            print("No chunk files found. Run chunker first.")
            return 0

        total_added = 0
        for chunk_file in chunk_files:
            with open(chunk_file) as f:
                chunks = json.load(f)
            added = self.add_chunks(chunks)
            print(f"  {chunk_file.name}: {added} new chunks indexed")
            total_added += added

        return total_added

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self, text: str, n_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Semantic search. Returns list of:
          {chunk_id, text, chapter, page_num, source_url, distance}
        """
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            output.append(
                {
                    "chunk_id": cid,
                    "text": doc,
                    "chapter": meta.get("chapter", ""),
                    "page_num": meta.get("page_num", 0),
                    "source_url": meta.get("source_url", ""),
                    "distance": dist,
                }
            )

        return output

    def count(self) -> int:
        return self.collection.count()
