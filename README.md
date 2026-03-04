# Medicare Policy Q&A with Hybrid GraphRAG

**Author:** Sreya Kavil Kamparath

---

## Problem Statement - Why This Exists

Anyone who has tried to navigate CMS Medicare policy documentation knows the pain. The manuals are dense, deeply cross-referenced, and written in policy-speak. A simple question like *"Does Medicare cover home health visits after a hospitalization, and what are the eligibility requirements?"* might require reading across three different chapters, two subsections, and a handful of exception clauses — none of which link to each other in any usable way.

For healthcare administrators, billing teams, and compliance officers, this isn't a one-off problem. It's a daily one. They need answers that are accurate, traceable, and fast. A hallucinated answer from a standard chatbot isn't just unhelpful — it can lead to incorrect claims, audit failures, or denied coverage.

This project is my attempt to build something that actually solves this well.

---

## What I Built

A hybrid GraphRAG system that ingests CMS Medicare policy PDFs and lets users ask natural language questions against them — with cited, traceable answers.

The system combines three things that are usually treated separately:

- A **vector database** (ChromaDB) for semantic similarity search over policy text chunks
- A **knowledge graph** (Neo4j) of entities and relationships extracted from the same text by Claude
- A **Claude Sonnet** answer layer that synthesizes both sources and cites every claim by chapter and page number

The result is a Streamlit Q&A interface where answers include inline citations like `[Source: Chapter 01, p.34]` and a sidebar panel with source excerpts for verification.

---

## The Technical Tradeoffs: Why Not Just RAG?

This is worth explaining in some depth, because the choice of architecture reflects a deliberate set of product decisions.

### Traditional RAG (Retrieval-Augmented Generation)

Standard RAG works like this: embed your documents into a vector store, embed the user query, find the closest chunks by cosine similarity, and pass them to an LLM.

It works well when the answer lives in one place. But policy documents aren't like that. Coverage for a service might be defined in one section, the eligibility criteria in another, and the exclusions somewhere else entirely. Vector similarity finds what *sounds like* the query — not necessarily what's *connected* to it.

**Failure mode:** A user asks about Home Health coverage. The vector search surfaces three paragraphs about home health. None of them mention that skilled nursing is a prerequisite, because that's defined separately. The LLM gives a confident but incomplete answer.

### Graph RAG (Knowledge Graph Only)

Pure Graph RAG extracts entities and relationships from your corpus and stores them as a graph. Retrieval becomes graph traversal — follow the edges from a starting node to discover connected concepts.

This is excellent for relationship-heavy questions: *"What services does Medicare exclude from SNF coverage?"* But it struggles with nuanced policy language, hedged conditions ("unless the beneficiary has..."), or anything that requires reading the actual text rather than summarized triples.

**Failure mode:** The graph knows that `Skilled Nursing Facility --[EXCLUDES]--> Custodial Care` but loses the *why* — the seven specific conditions that define custodial care. Answer quality degrades on anything requiring verbatim policy interpretation.

### Hybrid RAG (What This System Does)

The retrieval pipeline here runs both in sequence:

1. Vector search over ChromaDB returns the top-5 semantically relevant chunks
2. Claude performs lightweight NER on those chunks to extract entity names
3. Those entity names are used to query Neo4j — pulling 1-hop neighbors for each
4. The merged result (text chunks + graph triples) is passed to Claude Sonnet as a unified context window

This means the answer model sees both the raw policy text *and* the structured relationship map around it. The graph fills in cross-references that the vector search would have missed. The text fills in nuance that the graph can't represent.

It's not a perfect system — no system is — but it's a much better match for the structure of the actual problem.

---

## Architecture

```
CMS PDFs
    │
    ▼
[Chunker]  ──── 800-token windows, 150-token overlap ────► chunks.json
    │
    ▼
[Extractor]  ── Claude Haiku (NER + relation extraction) ─► extracted.json
    │
    ├──► [Neo4j]   Entity nodes + typed relationship edges
    │              (REQUIRES, COVERS, EXCLUDES, REFERENCES, DEFINES, APPLIES_TO)
    │
    └──► [ChromaDB] all-MiniLM-L6-v2 sentence embeddings
                    of raw policy chunks

                         Query
                           │
                    [Hybrid Retriever]
                    │              │
              Vector Search    Graph Traversal
              (ChromaDB)       (Neo4j 1-hop)
                    │              │
                    └──────┬───────┘
                           │
                    [Claude Sonnet]
                    Answer synthesis + citation
                           │
                    [Streamlit UI]
                    Chat + source panel
```

**Entity types extracted:** Service, Condition, Coverage, Requirement, Term, Policy

**Models used:**
- `claude-haiku-4-5-20251001` — extraction and NER (cost-efficient, structured output)
- `claude-sonnet-4-6` — answer generation (reasoning quality, citation accuracy)

---

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Anthropic Claude (Haiku + Sonnet) |
| Vector DB | ChromaDB |
| Graph DB | Neo4j 5.15 Community (Docker) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| PDF parsing | PyPDF2 |
| UI | Streamlit |
| Orchestration | Python, Docker Compose |

---

## Getting Started

**Prerequisites:** Docker, Python 3.11+, Anthropic API key

```bash
# 1. Clone and configure
git clone <this-repo>
cd medical-graphrag
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, NEO4J_PASSWORD, CHROMA_PERSIST_DIR

# 2. Start Neo4j
docker-compose up -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download source PDFs (CMS Chapters 01 and 15)
python scripts/download_pdfs.py

# 5. Build the full index (chunk → extract → graph → embed)
python scripts/build_index.py

# 6. Launch the app
streamlit run src/app.py
```

The index build runs all five pipeline stages in sequence and takes ~20–40 minutes depending on API rate limits. Progress is logged to console.

---

## Project Structure

```
├── data/
│   ├── pdfs/              # Raw CMS policy PDFs
│   └── processed/         # Chunks and extracted entities (JSON)
├── scripts/
│   ├── download_pdfs.py   # Fetch CMS source documents
│   └── build_index.py     # Full pipeline orchestrator
├── src/
│   ├── ingestion/
│   │   ├── chunker.py     # PDF → overlapping text chunks
│   │   └── extractor.py   # Claude Haiku entity/rel extraction
│   ├── graph/
│   │   ├── neo4j_client.py
│   │   └── builder.py     # Load extracted JSON → Neo4j
│   ├── retrieval/
│   │   ├── vector_store.py # ChromaDB wrapper
│   │   └── hybrid.py      # Combined retrieval pipeline
│   ├── qa/
│   │   └── answerer.py    # Claude Sonnet answer + citations
│   └── app.py             # Streamlit UI
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## This Problem Isn't Just Healthcare

The core challenge here — dense, cross-referenced policy documents that require multi-hop reasoning to answer accurately — shows up in many industries:

**Financial Services**
Regulatory compliance teams at banks deal with the same structural problem. Fed/CFPB rulebooks, Basel III capital requirements, or Dodd-Frank provisions aren't organized for lookup — they're organized for sequential reading. A compliance officer asking *"Which reporting requirements apply to loans originated under Program X if the borrower is below the HMDA threshold?"* needs exactly this kind of multi-hop retrieval. The same pipeline with different entity types (Regulation, Threshold, Exemption, Requirement) would work directly.

**IT and Information Security**
Enterprise IT teams managing vendor contracts, SLA agreements, and security compliance frameworks (SOC 2, ISO 27001) face similar navigation problems. Which controls apply to which systems under which exception clauses? A GraphRAG system over your compliance documentation would let security engineers query their audit requirements the same way a lawyer queries case law.

**Customer Support at Scale**
Large enterprises with complex product catalogs — telecom, insurance, utilities — maintain thousands of pages of internal policy documentation that support agents have to interpret in real time. The failure mode when agents can't find the right answer is either a wrong answer or an unnecessarily escalated ticket. A hybrid retrieval system over internal knowledge bases would surface cross-referenced product rules that standard keyword search would never find.

**Legal and Contracting**
Contract review is inherently relationship-heavy: Party A's obligation in Clause 12 may be conditioned on definitions in Clause 3 and exceptions in an exhibit. Building entity graphs over contract corpora could make contract Q&A far more reliable than embedding the full document and hoping the vector search lands on the right clause.

---

## What I'd Build Next

**1. OCR-based parsing for scanned PDFs**
CMS policy documents are often distributed as scanned image PDFs, especially older chapters and archived versions. PyPDF2 can't extract text from images. Adding a Tesseract or AWS Textract OCR stage before chunking would make the pipeline work on the full document corpus, not just the chapters that happen to have embedded text layers.

**2. Cross-chapter relationship inference**
Right now, entity extraction and graph building happen per-chapter. Entities with the same name across chapters aren't explicitly linked — the graph has two separate `Skilled Nursing Facility` nodes, one per chapter, rather than a unified concept node with edges pointing to both source documents. A deduplication and entity resolution pass would allow genuinely cross-document multi-hop queries: *"How does the definition of 'medical necessity' in Chapter 1 compare to its application in Chapter 15?"*

**3. Answer confidence scoring and uncertainty signaling**
The system currently tells the user *what* the policy says but not *how confident* it is. For high-stakes compliance questions, that matters. A confidence layer — based on the number of corroborating chunks, graph path length, and whether the graph and vector results agreed or diverged — would let users know when to go verify the source directly rather than trusting the summary. This is a product feature, not just a ML one: the UI should communicate epistemic uncertainty in a way non-technical users can act on.

---

## Notes on Cost

Running the full extraction pipeline on two CMS chapters (Chapters 01 and 15) costs roughly $1.50–$2.00 in Anthropic API credits using Claude Haiku for extraction. Answer generation with Claude Sonnet is approximately $0.002–$0.005 per query depending on context length. The pipeline is designed to run once offline; queries are cheap.

---

*Built as part of a personal exploration into RAG architectures for complex document domains. All policy documents used are publicly available from CMS.gov.*
