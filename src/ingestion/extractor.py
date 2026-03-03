"""
Entity + relationship extraction from text chunks using Claude.
Uses claude-haiku-4-5-20251001 for cost efficiency.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 4
BASE_DELAY = 2.0  # seconds

SYSTEM_PROMPT = """You are a medical policy knowledge graph extractor.
Extract entities and relationships from Medicare policy text and return ONLY valid JSON.
No prose, no markdown fences, just the JSON object."""

EXTRACTION_PROMPT = """Extract entities and relationships from this Medicare policy text.

Entity types (use exactly these): Service, Condition, Coverage, Requirement, Term, Policy

Relationship types (use exactly these): REQUIRES, COVERS, EXCLUDES, REFERENCES, DEFINES, APPLIES_TO

Return ONLY this JSON structure (no markdown, no extra text):
{{
  "entities": [
    {{"name": "entity name", "type": "EntityType", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "entity name", "relation": "RELATION_TYPE", "target": "entity name", "evidence": "quote from text"}}
  ]
}}

Rules:
- Entity names must be concise (2-6 words max)
- Only include relationships between entities you listed
- Evidence should be a short direct quote (under 20 words)
- Return empty arrays if nothing relevant is found

TEXT:
{text}"""


def _call_claude(client: anthropic.Anthropic, text: str) -> dict[str, Any]:
    """Call Claude with retry on rate limits / transient errors."""
    prompt = EXTRACTION_PROMPT.format(text=text[:3000])  # cap to ~750 tokens input

    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

            # Strip markdown fences if Claude adds them despite instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"      JSON parse error (attempt {attempt + 1}): {e}")
            if attempt == MAX_RETRIES - 1:
                return {"entities": [], "relationships": []}

        except anthropic.RateLimitError:
            delay = BASE_DELAY * (2 ** attempt)
            print(f"      Rate limit hit. Waiting {delay:.1f}s...")
            time.sleep(delay)

        except anthropic.APIError as e:
            delay = BASE_DELAY * (2 ** attempt)
            print(f"      API error (attempt {attempt + 1}): {e}. Waiting {delay:.1f}s...")
            time.sleep(delay)

    return {"entities": [], "relationships": []}


def extract_from_chunks(
    chunks: list[dict[str, Any]],
    chapter: str,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """
    Run entity/relationship extraction on each chunk.
    Returns list of extraction results with chunk metadata attached.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks), chunk["chunk_id"])

        extraction = _call_claude(client, chunk["text"])

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "chapter": chunk["chapter"],
                "page_num": chunk["page_num"],
                "source_url": chunk["source_url"],
                "entities": extraction.get("entities", []),
                "relationships": extraction.get("relationships", []),
            }
        )

        # Small polite delay to avoid hammering the API
        time.sleep(0.3)

    return results


def save_extracted(results: list[dict[str, Any]], chapter: str) -> Path:
    """Persist extraction results to data/processed/{chapter}_extracted.json."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{chapter}_extracted.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved → {output_path}")
    return output_path


def process_chapter(chapter: str) -> Path | None:
    """Load chunks for a chapter, extract, and save."""
    chunks_path = PROCESSED_DIR / f"{chapter}_chunks.json"
    if not chunks_path.exists():
        print(f"  Chunks file not found: {chunks_path}")
        return None

    with open(chunks_path) as f:
        chunks = json.load(f)

    print(f"  Extracting from {len(chunks)} chunks in {chapter}...")

    def progress(i, total, chunk_id):
        if i % 10 == 0 or i == total - 1:
            print(f"    [{i + 1}/{total}] Processing {chunk_id}")

    results = extract_from_chunks(chunks, chapter, progress_callback=progress)

    entity_count = sum(len(r["entities"]) for r in results)
    rel_count = sum(len(r["relationships"]) for r in results)
    print(f"    Extracted {entity_count} entities, {rel_count} relationships")

    return save_extracted(results, chapter)


def process_all_chapters() -> list[Path]:
    """Extract from all chunk files in data/processed/."""
    chunk_files = sorted(PROCESSED_DIR.glob("*_chunks.json"))
    if not chunk_files:
        print("No chunk files found. Run chunker first.")
        return []

    output_paths = []
    for chunk_file in chunk_files:
        chapter = chunk_file.stem.replace("_chunks", "")
        out = process_chapter(chapter)
        if out:
            output_paths.append(out)

    return output_paths


if __name__ == "__main__":
    process_all_chapters()
