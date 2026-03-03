"""
PDF → overlapping text chunks with metadata.
Window: 800 tokens (~600 words), overlap: 150 tokens (~112 words).
"""

import json
import re
from pathlib import Path
from typing import Any

import PyPDF2

# Approximate chars-per-token ratio for English text
CHARS_PER_TOKEN = 4
WINDOW_TOKENS = 800
OVERLAP_TOKENS = 150
WINDOW_CHARS = WINDOW_TOKENS * CHARS_PER_TOKEN   # ~3200
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN  # ~600

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove PDF extraction artifacts."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text from each page of a PDF."""
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = _clean_text(text)
            if text:
                pages.append({"page_num": page_num, "text": text})
    return pages


def chunk_pages(
    pages: list[dict[str, Any]],
    chapter: str,
    source_url: str,
) -> list[dict[str, Any]]:
    """
    Merge all page text into a single string, then slide a window over it.
    Each chunk carries the page number where it started.
    """
    # Build a flat text with page boundary markers
    full_text = ""
    page_offsets: list[tuple[int, int]] = []  # (char_offset, page_num)
    for p in pages:
        page_offsets.append((len(full_text), p["page_num"]))
        full_text += p["text"] + " "

    def _page_at(offset: int) -> int:
        """Return the page number that contains character offset."""
        page_num = 1
        for char_off, pnum in page_offsets:
            if char_off <= offset:
                page_num = pnum
            else:
                break
        return page_num

    chunks = []
    chunk_id = 0
    start = 0

    while start < len(full_text):
        end = min(start + WINDOW_CHARS, len(full_text))
        text_slice = full_text[start:end].strip()

        if len(text_slice) < 50:  # skip tiny trailing slices
            break

        chunks.append(
            {
                "chunk_id": f"{chapter}_{chunk_id:04d}",
                "chapter": chapter,
                "page_num": _page_at(start),
                "source_url": source_url,
                "text": text_slice,
                "char_start": start,
                "char_end": end,
            }
        )
        chunk_id += 1
        start += WINDOW_CHARS - OVERLAP_CHARS  # slide forward with overlap

    return chunks


def chunk_pdf(pdf_path: Path, source_url: str = "") -> list[dict[str, Any]]:
    """Full pipeline: PDF → cleaned pages → overlapping chunks."""
    chapter = pdf_path.stem  # e.g. "chapter_01"
    print(f"  Chunking {pdf_path.name}...")

    pages = extract_pages(pdf_path)
    print(f"    {len(pages)} pages extracted")

    chunks = chunk_pages(pages, chapter, source_url)
    print(f"    {len(chunks)} chunks produced")

    return chunks


def save_chunks(chunks: list[dict[str, Any]], chapter: str) -> Path:
    """Persist chunks to data/processed/{chapter}_chunks.json."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{chapter}_chunks.json"
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"    Saved → {output_path}")
    return output_path


def process_all_pdfs() -> list[Path]:
    """Chunk all PDFs in data/pdfs/ and save results."""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/pdfs/. Run scripts/download_pdfs.py first.")
        return []

    output_paths = []
    for pdf_path in pdf_files:
        source_url = (
            f"https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/{pdf_path.name}"
        )
        chunks = chunk_pdf(pdf_path, source_url)
        out = save_chunks(chunks, pdf_path.stem)
        output_paths.append(out)

    return output_paths


if __name__ == "__main__":
    process_all_pdfs()
