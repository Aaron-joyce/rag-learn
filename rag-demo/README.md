# Agentic RAG Demo

A Retrieval-Augmented Generation system where the LLM decides *when* and *what* to retrieve, rather than retrieval happening on a fixed pipeline before every query. Built to understand how agentic tool-calling changes RAG behavior compared to naive "always retrieve, always stuff into context" implementations.

## How It Works

Most basic RAG tutorials retrieve on every query regardless of whether retrieval is needed. This implementation instead exposes retrieval as **tools** the model can choose to call:

- `search_knowledge_base(query)` — semantic search over the indexed document chunks
- `list_available_documents()` — lets the model check what's actually in the knowledge base before deciding how to answer

Gemini (`gemini-3.5-flash`) receives these as native function-calling tools and decides during generation whether to invoke them, inspect the results, and then produce a grounded answer with source citations — rather than retrieval being hardcoded into every request.

## Architecture

```
Query → Gemini (gemini-3.5-flash)
              │
              ├─ decides to call search_knowledge_base()
              │        │
              │        ▼
              │   Chroma DB (persistent, local)
              │        │
              │        ▼
              │   top-k chunks (Cosine HNSW search)
              │
              ▼
        Grounded answer + cited sources
```

**Indexing (offline, run once per document set):**
1. Documents in `docs/` are loaded and split with a custom sentence-aware chunker (`chunk_text`) — ~500-character chunks with 50-character overlap, splitting on sentence boundaries (`. `, `? `, `\n`) rather than hard character cutoffs, so chunks don't get sliced mid-sentence.
2. Each chunk is embedded locally using `sentence-transformers` (`all-MiniLM-L6-v2`) — no API call needed for embeddings.
3. Chunks + embeddings are upserted into a persistent Chroma collection (`./chroma_db`).

**Query time:**
1. User question goes to `gemini-3.5-flash` along with the two tool definitions.
2. The model decides whether the question needs retrieval, and if so, calls `search_knowledge_base` with a query it constructs itself.
3. Chroma returns the top-k most similar chunks (cosine similarity, HNSW index).
4. The model reads the retrieved chunks and generates an answer, citing which source chunk(s) it used.

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| LLM | Gemini `gemini-3.5-flash` via `google-genai` | Native function-calling support, fast/cheap enough for iterative testing |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Runs locally — no per-chunk API cost or latency during indexing |
| Vector store | ChromaDB (persistent, local) | Zero-ops local vector DB, good enough for a single-machine demo |
| Env/package management | `uv` | Fast, lockfile-based reproducible installs |

## Running It

**Requirements:** Python 3.14, [`uv`](https://docs.astral.sh/uv/) installed, a `GEMINI_API_KEY`.

```bash
cd rag-demo
uv sync                          # installs from uv.lock
cp .env.example .env             # add your GEMINI_API_KEY here
uv run basic_rag_agent.py        # indexes docs/ and starts the query loop
```

On first run, it indexes everything under `docs/` into `./chroma_db`. Add your own markdown/text files there to query different content.

## Design Notes / Trade-offs

- **Tool-calling retrieval vs. always-retrieve:** giving the model the choice to retrieve (and to call `list_available_documents` first) means it can reason about what's actually available before searching, instead of blindly retrieving top-k chunks for every query — including ones that don't need retrieval at all.
- **Local embeddings over API embeddings:** avoids per-chunk API costs and network latency during indexing, at the cost of lower embedding quality than a larger hosted model. Fine for a small demo corpus; would reconsider for a larger or more nuanced knowledge base.
- **Sentence-aware chunking:** naive fixed-character chunking risks splitting a sentence in half across two chunks, which can hurt retrieval relevance. Respecting sentence boundaries costs a bit of complexity in the chunker for better chunk coherence.

## Known Limitations

- Single-machine, local-only — no deployment/serving layer (e.g. no API wrapper around the agent loop yet)
- Small demo corpus (`docs/api.md`) — not yet tested against a larger or more heterogeneous document set
- No conversation memory across turns — each query is currently independent

## What I'd Build Next

- Wrap the agent loop in a minimal API (FastAPI) so it's callable as a service, not just a terminal script
- Add conversation history so follow-up questions can reference prior turns
- Evaluate retrieval quality more rigorously (e.g. a small labeled query set) rather than just eyeballing answers
