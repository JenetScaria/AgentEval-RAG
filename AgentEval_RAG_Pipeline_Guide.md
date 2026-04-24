# AgentEval RAG — Complete Pipeline Guide

> **Who is this for?**
> This guide explains the entire AgentEval RAG system from scratch. No prior experience with RAG, LLMs, or AI pipelines assumed. Every concept is explained before it is used.

---

## What is this system?

Imagine you have thousands of Amazon product reviews and you want to ask questions like:

> *"What do customers say about HEPA filters for dust allergies?"*

A normal search engine would show you keyword matches. This system does something smarter:

1. It **understands the meaning** of your question
2. It **finds the most relevant reviews** from its database
3. It **generates a proper written answer** with citations
4. It **grades its own answer** and **tries again** if the answer wasn't good enough

That last part — grading itself and retrying — is what makes it "agentic" (the system makes decisions on its own) and "self-evaluating."

---

## The Big Picture

Here is the full flow from the moment you type a question to the moment you get an answer:

```
You type a question
        │
        ▼
┌───────────────┐
│  1. ROUTER    │  ← "What kind of question is this?"
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  2. RETRIEVAL │  ← "Find the most relevant text chunks"
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  3. GENERATION│  ← "Write an answer based on those chunks"
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  4. EVALUATION│  ← "How good was that answer? Score it."
└───────┬───────┘
        │
   Good score?
   ┌────┴────┐
  YES       NO
   │         │
   ▼         ▼
Show      ┌──────────────┐
answer    │ 5. RETRY     │ ← Try again, maybe search the web
          └──────────────┘
```

These 5 steps are implemented as **nodes in a LangGraph graph** — think of it as a flowchart written in code, where each box is a Python function.

---

## Before You Can Ask Questions — Ingestion

Before the system can answer anything, it needs to **read and index** all the Amazon reviews. This is a one-time setup step done by running `ingest.py`.

### Step 1 — Download the data

```
McAuley-Lab/Amazon-C4 dataset (HuggingFace)
           │
           │  streaming (one row at a time, no huge download)
           ▼
   Each row has:
   - query      : "I need filters that trap dust"
   - ori_review : "These filters work great, very impressed..."
           │
           ▼
   Combined into one text:
   "Q: I need filters that trap dust
    Review: These filters work great..."
```

**Why combine query + review?** Because when someone searches for "dust filter", they might use completely different words than the review. Combining both gives the retrieval system more vocabulary to match against.

Each combined text becomes a **Document** — LangChain's wrapper that holds text + metadata (like the product ID and star rating).

### Step 2 — Chunking

Reviews can be very long. Long text is harder to search and harder for an AI to process. So every Document gets cut into smaller **chunks** of 512 characters with a 50-character overlap.

```
Original review (800 chars):
"These filters work great I could not believe the amount of dust
 in the old filter when I replaced it after just one month. The
 air in my home feels cleaner and my allergies have improved..."

After chunking:
  Chunk 1 (512 chars): "These filters work great... allergies have"
  Chunk 2 (512 chars): "...allergies have improved. I would recommend"
                              ↑
                         50-char overlap so no sentence
                         gets cut off between chunks
```

**Why overlap?** If a key sentence falls exactly at a boundary, it would be split in half. The overlap ensures it appears fully in at least one chunk.

### Step 3 — Build the FAISS Index (Meaning Search)

Every chunk of text gets converted into a **vector** — a list of 384 numbers that represent the *meaning* of the text.

```
"These filters trap dust"  →  [0.12, -0.34, 0.87, 0.03, ...]  (384 numbers)
"Air purifier works well"  →  [0.11, -0.31, 0.85, 0.04, ...]  ← similar numbers = similar meaning
"My cat is orange"         →  [-0.9,  0.22, -0.11, 0.77, ...]  ← very different numbers
```

This conversion is done by **SentenceTransformer** (`all-MiniLM-L6-v2`), a small AI model specifically trained to turn sentences into meaningful numbers.

All these vectors are stored in a **FAISS index** — a library from Facebook that can search through millions of vectors in milliseconds to find the ones closest to your query vector.

**FAISS is great at:** finding text with similar *meaning*, even if the words are completely different.

### Step 4 — Build the BM25 Index (Keyword Search)

Every chunk is also broken into individual words and stored in a **BM25 index**.

```
"These filters trap dust"  →  ["these", "filters", "trap", "dust"]
```

BM25 is the algorithm that powers classic search engines. It scores documents based on how often your search words appear in them, adjusted for document length.

**BM25 is great at:** finding exact keywords, model numbers, brand names, rare terms.

### Step 5 — Save to Disk

Three files are saved to `data/processed/`:

| File | What it stores |
|---|---|
| `faiss_index.faiss` | All the meaning-vectors |
| `bm25_index.pkl` | The keyword index |
| `faiss_index_docs.pkl` | The original text of all chunks |

The docs file is crucial — FAISS only stores numbers, not text. When FAISS says "chunk #4231 is the best match", you need the docs file to get the actual review text back.

---

## The Pipeline — Step by Step

Now let's walk through what happens every time you ask a question.

---

### Node 1 — Router (`router_node` in `graph.py`)

**What it does:** Reads your question and decides what *type* of question it is.

**The three types:**

| Type | Meaning | Example |
|---|---|---|
| `simple` | One straightforward fact | "Do these filters remove pet dander?" |
| `multi_hop` | Needs info from multiple places | "Compare dust vs pet filter performance across brands" |
| `web` | Needs current/live information | "What is the latest HEPA filter technology in 2025?" |

**How it works:** It sends your question to **Gemini** (the free Google AI) with a strict instruction: *"Reply with one word only: simple, multi_hop, or web."*

```python
# Simplified version of what happens:
gemini("Do these filters remove pet dander?",
       system="Reply with one word: simple, multi_hop, or web")
# → "simple"
```

**Why classify?** Because `simple` questions use one fast search, `multi_hop` questions need two searches (first get some info, then search for more based on what you found), and `web` questions need to search the internet instead of the local database.

---

### Node 2 — Retrieval (`retrieval_node` in `graph.py`)

**What it does:** Finds the text chunks most likely to contain the answer.

This node has three paths depending on the query type from Node 1:

#### Path A — Simple retrieval (Hybrid Search)

This is the most common path. It runs both FAISS and BM25, combines their scores, then reranks.

**Step 1 — FAISS search (60% weight)**

Your question gets converted to a vector by the same SentenceTransformer model used during ingestion. FAISS finds the 10 chunks whose vectors are closest to your question vector.

```
Your question: "Which filters are best for dust?"
                        ↓
              Convert to vector: [0.15, -0.32, 0.88, ...]
                        ↓
              FAISS finds 10 closest vectors
                        ↓
              Returns: chunk #43 (score 0.92), chunk #817 (score 0.89), ...
```

**Step 2 — BM25 search (40% weight)**

Your question is split into words and BM25 finds the 10 chunks where those words appear most frequently.

```
Your question: "Which filters are best for dust?"
                        ↓
              Split: ["which", "filters", "are", "best", "for", "dust"]
                        ↓
              BM25 finds chunks with most of these words
                        ↓
              Returns: chunk #43 (score 0.85), chunk #201 (score 0.71), ...
```

**Step 3 — Score fusion**

Scores from both searches are combined: `final_score = (FAISS score × 0.6) + (BM25 score × 0.4)`

```
chunk #43:  (0.92 × 0.6) + (0.85 × 0.4) = 0.552 + 0.340 = 0.892  ✓ top result
chunk #817: (0.89 × 0.6) + (0.00 × 0.4) = 0.534 + 0.000 = 0.534
chunk #201: (0.00 × 0.6) + (0.71 × 0.4) = 0.000 + 0.284 = 0.284
```

**Step 4 — Cross-encoder reranking**

The top 10 fused results get re-evaluated by a **Cross-Encoder** — a smarter, slower model that reads your question and each chunk *together* and gives a more precise relevance score. The top 5 survive.

```
Cross-encoder reads:
  Question: "Which filters are best for dust?"
  + Chunk #43: "Q: I need filters that trap dust\nReview: These work great..."
→ Relevance score: 0.94

  Question: "Which filters are best for dust?"
  + Chunk #817: "Q: Air quality improvement\nReview: My home smells fresher..."
→ Relevance score: 0.61
```

The cross-encoder is slower but more accurate than FAISS alone — it understands context, not just vector similarity.

**Why both FAISS and BM25?**

| Scenario | FAISS alone | BM25 alone | Both |
|---|---|---|---|
| "dust removal performance" | Finds "particulate filtration" ✓ | Misses it (no exact words) ✗ | ✓ |
| "model B0C5QYYHTJ review" | Might miss exact code ✗ | Finds exact code ✓ | ✓ |

#### Path B — Multi-hop retrieval

Runs the hybrid search **twice** with different queries:

1. **Hop 1:** Search with original question → get initial results
2. **Gemini generates a follow-up query** based on what was found
3. **Hop 2:** Search again with the follow-up query → get more targeted results
4. Merge all unique results

#### Path C — Web search

Tries **Tavily** (paid, more accurate) first, then falls back to **DuckDuckGo** (free) if Tavily isn't configured. Returns live web results instead of local chunks.

---

### Node 3 — Generation (`generation_node` in `graph.py`)

**What it does:** Takes the retrieved chunks and writes a proper answer.

The retrieved chunks (up to 5) are formatted with numbers:

```
[1] Q: I need filters for dust
    Review: These filters work great, the old filter was full of dust after one month...

[2] Q: Air quality at home
    Review: I could see a difference in the air quality within days...

[3] ...
```

This numbered context + your question is sent to **Gemini** with a strict instruction:

> *"Answer based strictly on the provided numbered context. Cite sources inline using [N] notation."*

Gemini writes an answer like:
> "Based on customer reviews, these filters are highly effective at trapping dust [1]. Many customers noticed visible dust accumulation in their old filters after just one month of use [1], and reported improved air quality within days of installation [2]."

The `[1]` and `[2]` in the answer are extracted to track which chunks were actually used — these become the **citations** you see in the UI.

---

### Node 4 — Evaluation (`eval_node` in `graph.py`)

**What it does:** Automatically grades the answer on three dimensions.

This is the "self-evaluating" part that makes this system special. Most RAG systems just return an answer and hope for the best. This one measures quality.

#### The three metrics

**1. Faithfulness** — Did the answer only use information from the retrieved chunks?

```
Retrieved chunk says: "filters trap dust"
Answer says:          "filters trap dust and also purify viruses"
                                                    ↑
                                    This wasn't in the chunks!
Faithfulness score: LOW (answer made something up)
```

**2. Context Precision** — Were the retrieved chunks actually relevant to the question?

```
Question: "Are these good for dust?"
Retrieved chunk: "Product ships in eco-friendly packaging"
                  ↑
                  Completely irrelevant!
Context Precision: LOW (retrieved wrong chunks)
```

**3. Answer Relevancy** — Does the answer actually address the question?

```
Question: "Are these good for dust?"
Answer:   "The product was developed in 2019 by a company in Texas."
           ↑
           Didn't answer the question at all!
Answer Relevancy: LOW
```

All three scores are between 0 and 1. The thresholds (set in `config.py`) are 0.7 for all three — so a score below 0.7 on any metric triggers a retry.

#### How are scores calculated?

The system first tries **RAGAS** (a dedicated AI evaluation library). If RAGAS isn't available or errors out, it falls back to **heuristic scoring** — simple word-overlap calculations:

- Faithfulness ≈ "how many words in the answer also appear in the retrieved chunks?"
- Context Precision ≈ "how many question words appear in the retrieved chunks?"
- Answer Relevancy ≈ "how many question words appear in the answer?"

These heuristics are rough approximations — RAGAS scores are much more reliable — but they work as a fallback so the pipeline never crashes.

#### MLflow logging

Every query's scores, parameters, and metadata are logged to **MLflow** (a tool for tracking ML experiments). You can open the MLflow UI with:

```bash
mlflow ui
```

and see a history of every query, its scores, whether it retried, and whether web search was used.

---

### Node 5 — Retry / Re-retrieval (`reretrieval_node` in `graph.py`)

**What it does:** Decides how to try again when scores are too low.

This node only runs if evaluation scores are below threshold AND the system hasn't already retried `max_retries` times (default: 2).

**Retry strategy:**
- **First retry** → Switch to web search (maybe the answer isn't in the local database)
- **Second retry** → Go back to local database search

After re-retrieval, the pipeline loops back to Generation → Evaluation again.

```
First attempt:  local search → answer → score 0.45 (too low) → retry
Second attempt: web search   → answer → score 0.82 (good!)   → show answer
```

If after all retries the score is still low, the best answer found is returned anyway.

---

## The Streamlit UI (`app.py`)

The web interface is built with **Streamlit** — a Python library for building simple web apps.

**What you see:**
- **Text input** — where you type your question
- **Answer section** — the generated answer with markdown formatting
- **Citations expander** — which chunks were used to write the answer
- **Metadata row** — query type (Simple/Multi Hop/Web), retry count, web fallback used
- **Evaluation scores** — faithfulness, context precision, answer relevancy, overall
- **Overall quality indicator** — green (≥0.7), orange (≥0.5), red (<0.5)

**Sidebar** shows the current threshold settings from `config.py`.

---

## Configuration (`config.py`)

All tunable settings live in one place and are loaded from the `.env` file:

| Setting | Default | What it controls |
|---|---|---|
| `GEMINI_API_KEY` | — | Your Google AI API key (required) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Which Gemini model to use |
| `CHUNK_SIZE` | `512` | How many characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `10` | How many chunks to retrieve before reranking |
| `TOP_K_RERANK` | `5` | How many chunks survive reranking |
| `FAITHFULNESS_THRESHOLD` | `0.7` | Minimum faithfulness score (0–1) |
| `CONTEXT_PRECISION_THRESHOLD` | `0.7` | Minimum context precision score (0–1) |
| `ANSWER_RELEVANCY_THRESHOLD` | `0.7` | Minimum answer relevancy score (0–1) |
| `MAX_RETRIES` | `2` | How many times to retry before giving up |

---

## File Map

```
agenteval_rag/
│
├── config.py                        ← All settings (reads from .env)
├── graph.py                         ← The 5-node LangGraph pipeline
├── app.py                           ← Streamlit web UI
├── api.py                           ← FastAPI REST API (alternative to UI)
│
├── src/
│   ├── retrieval/
│   │   └── hybrid_retriever.py      ← FAISS + BM25 + cross-encoder
│   ├── evaluation/
│   │   └── ragas_eval.py            ← RAGAS scorer + heuristic fallback
│   └── utils/
│       ├── ingest.py                ← One-time data pipeline (run first!)
│       └── mlflow_logger.py         ← Experiment tracking
│
├── data/
│   └── processed/
│       ├── faiss_index.faiss        ← Generated by ingest.py
│       ├── bm25_index.pkl           ← Generated by ingest.py
│       └── faiss_index_docs.pkl     ← Generated by ingest.py
│
├── tests/
│   └── test_retrieval.py            ← pytest tests (no disk indexes needed)
│
├── .env                             ← Your API keys and settings
└── requirements.txt                 ← Python dependencies
```

---

## How to Run (Quick Reference)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key in .env
GEMINI_API_KEY=AIza...your_key_here

# 3. Build the indexes (one time only)
python src/utils/ingest.py --samples 5000

# 4. Start the app
streamlit run app.py
```

---

## Glossary

| Term | Plain English |
|---|---|
| RAG | Retrieval-Augmented Generation — find relevant text, then generate an answer from it |
| Agentic | The system makes its own decisions (retry, switch to web, etc.) |
| Vector / Embedding | A list of numbers that represents the meaning of a sentence |
| FAISS | Facebook's library for fast similarity search over vectors |
| BM25 | Classic keyword search algorithm used in search engines |
| Cross-encoder | A precise but slower relevance scorer that reads query + document together |
| RAGAS | A library for evaluating RAG system quality |
| LangGraph | A library for building multi-step AI pipelines as graphs |
| Chunk | A small piece of text split from a larger document |
| Heuristic | A simple rule-of-thumb approximation (used as fallback when exact method fails) |
| MLflow | A tool for logging and tracking ML experiments over time |
