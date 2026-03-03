"""
Answer generation using Claude Sonnet with citations.
Formats graph context + chunk context, then calls Claude to produce
a cited answer in the format [Source: Chapter X, p.Y].
"""

import os
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

ANSWER_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a Medicare policy expert assistant.
Answer questions accurately based ONLY on the provided policy context.
Always cite your sources using the format [Source: Chapter X, p.Y] inline.
If the context does not contain enough information to answer, say so clearly.
Be concise, precise, and helpful."""

ANSWER_PROMPT = """Answer the following question using the policy context below.

QUESTION: {question}

=== GRAPH KNOWLEDGE (entity relationships) ===
{graph_context}

=== POLICY TEXT (ranked by relevance) ===
{chunks_context}

Instructions:
- Synthesize information from both graph and text sources
- Cite every factual claim with [Source: Chapter X, p.Y]
- If sources contradict, note the discrepancy
- Keep your answer under 400 words unless complexity requires more"""


def _format_graph_context(graph_rows: list[dict[str, Any]]) -> str:
    """Format graph traversal results as readable triples."""
    if not graph_rows:
        return "(No graph relationships found)"

    lines = []
    for row in graph_rows[:15]:  # cap for prompt length
        src = row.get("source", "?")
        rel = row.get("relation", "?")
        tgt = row.get("target", "?")
        evidence = row.get("evidence", "")
        line = f"• {src} --[{rel}]--> {tgt}"
        if evidence:
            line += f'\n  Evidence: "{evidence}"'
        lines.append(line)

    return "\n".join(lines)


def _format_chunks_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks with source labels."""
    if not chunks:
        return "(No relevant text found)"

    lines = []
    for chunk in chunks:
        chapter = chunk.get("chapter", "unknown").replace("_", " ").title()
        page = chunk.get("page_num", "?")
        text = chunk.get("text", "").strip()
        label = f"[{chapter}, p.{page}]"
        lines.append(f"{label}\n{text[:800]}")  # cap per chunk
        lines.append("---")

    return "\n".join(lines)


def _parse_sources(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build structured source list from chunks for the UI citation panel."""
    sources = []
    seen = set()
    for chunk in chunks:
        chapter = chunk.get("chapter", "unknown").replace("_", " ").title()
        page = chunk.get("page_num", "?")
        key = (chapter, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "label": f"{chapter}, p.{page}",
                "chapter": chapter,
                "page_num": str(page),
                "source_url": chunk.get("source_url", ""),
                "text": chunk.get("text", "")[:500],
            }
        )
    return sources


class Answerer:
    def __init__(self, anthropic_client: anthropic.Anthropic | None = None):
        self.client = anthropic_client or anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    def answer(
        self,
        query: str,
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a cited answer from retrieval results.

        Args:
            query: The user's question
            retrieval_result: Output from HybridRetriever.retrieve()

        Returns:
            {answer: str, sources: [{label, chapter, page_num, text, source_url}]}
        """
        chunks = retrieval_result.get("chunks", [])
        graph_context = retrieval_result.get("graph_context", [])

        graph_text = _format_graph_context(graph_context)
        chunks_text = _format_chunks_context(chunks)

        prompt = ANSWER_PROMPT.format(
            question=query,
            graph_context=graph_text,
            chunks_context=chunks_text,
        )

        message = self.client.messages.create(
            model=ANSWER_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text = message.content[0].text.strip()
        sources = _parse_sources(chunks)

        return {
            "answer": answer_text,
            "sources": sources,
        }
