# TriageGraph — Support Ticket Triage Graph

An agentic customer-support pipeline built with **LangGraph**, **LangChain**, **Groq** and a **hybrid RAG** retriever.
A raw customer ticket goes in, a grounded, reviewed, human-approved reply email comes out.

```text
Ticket → Classify → Write search query → Hybrid retrieve → Draft reply → Grade → Human approve/edit/reject → Final
```

---

## Table of contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Configuration](#configuration)
- [Notes and limitations](#notes-and-limitations)
- [License](#license)

---

## Overview

`backend.py` compiles a LangGraph `StateGraph` that automates the first line of support:

1. **Classify** the ticket into one of `billing`, `technical`, `account`, `shipping`, `general` using a structured-output LLM call.
2. **Rewrite** the ticket into a short, PII-free knowledge-base search query.
3. **Retrieve** context from a company policy PDF with a hybrid retriever (dense vectors + BM25 + Cohere re-ranking).
4. **Draft** a reply email (`subject` + `body`) grounded only in the retrieved context.
5. **Grade** the draft with a strict LLM reviewer (relevant, grounded, actionable, tone, clean).
6. **Human-in-the-loop**: the graph pauses with `interrupt()`, a person approves, edits with feedback, or rejects.

State is checkpointed in **PostgreSQL** (`PostgresSaver`), so every conversation survives restarts and each ticket runs under its own `thread_id`.

---

## How it works

### Graph

```mermaid
flowchart LR
    START --> C[Classifier]
    C --> Q[Query_writer]
    Q --> R[Retriever]
    R --> D[Drafter]
    D --> G{Grader}
    G -- "no, retry_count < 3" --> D
    G -- yes --> H[Human_Approval]
    H -- edit, final empty --> D
    H -- approve / reject / limit reached --> END
```

### Nodes

| Node | File | What it does |
|------|------|--------------|
| `Classifier` | `backend.py:68` | Structured output → category (`classify_schema`) |
| `Query_writer` | `backend.py:109` | Ticket + category → one search query (≤ 25 words, no PII) |
| `Retriever` | `backend.py:150` | Runs `hybrid_search()` from `rag.py`, joins chunks into one context |
| `Drafter` | `backend.py:174` | Structured output → `subject` + `body` (`DraftEmail`), honours reviewer feedback |
| `Grader` | `backend.py:235` | Structured output → `yes`/`no` against 5 strict criteria |
| `Human_Approval` | `backend.py:300` | `interrupt()` → approve / edit (with feedback) / reject |

### State (`AgentState`)

`ticket`, `category`, `query`, `retrieved_context`, `draft`, `feedback`, `grader`, `approval`, `retry_count`, `edit_count`, `final`

### Guard rails

- Drafts are re-generated up to **3 times** while the grader answers `no`.
- A human can request up to **3 edits**; after that the last draft is returned with a limit notice.
- Prompts explicitly forbid following instructions written *inside* a ticket and forbid inventing policies, refunds or deadlines.

### Retriever (`rag.py`)

| Stage | Detail |
|-------|--------|
| Load | `PyPDFLoader` → `RecursiveCharacterTextSplitter` (1000 chars, 100 overlap) |
| Embeddings | `BAAI/bge-m3` (HuggingFace) |
| Vector store | Chroma, persisted in `./chroma_pdr`, collection `hybrid_demo` |
| Dense search | MMR (`k=5`, `fetch_k=10`) |
| Sparse search | BM25 (`k=5`) |
| Fusion | `EnsembleRetriever` weights `[0.5, 0.5]` |
| Re-rank | Cohere `rerank-v4.0-pro`, `top_n=6` via `ContextualCompressionRetriever` |

---

## Tech stack

- **Python 3.13**
- **LangGraph** — state graph, `interrupt()` human-in-the-loop, `langgraph-checkpoint-postgres`
- **LangChain** — prompts, structured output, document loaders, retrievers
- **Groq** — LLM (`qwen/qwen3.8-27b`) via `langchain-groq`
- **Cohere** — re-ranking via `langchain-cohere`
- **ChromaDB** + **BM25** — hybrid retrieval
- **PostgreSQL** + **psycopg 3** — checkpoint store
- **LangSmith** (optional) — tracing

---

## Project structure

```text
.
├── backend.py          # LangGraph agent, nodes, edges, interactive CLI loop
├── rag.py              # PDF loading, chunking, hybrid retriever (vector + BM25 + rerank)
├── test.py             # Standalone retrieval smoke test
├── main.py             # uv project placeholder
├── docs/               # Screenshots of real runs
├── chroma_pdr/         # Persisted Chroma vector store
├── .env                # API keys / DATABASE_URL (not committed)
├── requirements.txt
└── pyproject.toml
```

---

## Prerequisites

- Python **3.13**
- A **PostgreSQL** server (default connection used by the project: `postgresql://<user>:<pass>@localhost:5432/triage`)
- API keys:
  - [`GROQ_API_KEY`](https://console.groq.com/keys)
  - [`COHERE_API_KEY`](https://dashboard.cohere.com/api-keys)
  - optional `LANGSMITH_API_KEY` for tracing
- A company **policy PDF** (the knowledge base the agent answers from)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Yogeshwaran197/TriageGraph-Support-ticket-triage-graph-.git
cd TriageGraph-Support-ticket-triage-graph-

# 2. Create the environment (uv or venv + pip both work)
uv sync
# or: python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# 3. Create the database used for checkpointing
psql -U postgres -c "CREATE DATABASE triage;"

# 4. Create a .env file
```

`.env`:

```dotenv
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
DATABASE_URL=postgresql://postgres:password@localhost:5432/triage

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=triagegraph
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

> `backend.py` calls `checkpointer.setup()` on startup, so the checkpoint tables are created automatically — no manual migrations.

---

## Usage

### 1. Retrieval smoke test

```bash
python test.py
```

```text
Enter the path of policy pdf : C:\Users\you\Downloads\nimbus_triage_test_scenarios.pdf
Length of page : 5
Length of chunk : 10
question : How long is a password reset link valid and can it be used twice?

 result - 1 | Score - 0.9195631
 content : TEST N-003 ... Expected behavior: Account route. Policy: link valid 30 minutes, single use.
```

Type `exit` to quit.

### 2. Run the full agent

```bash
python backend.py
```

```text
Enter the path of policy pdf : C:\Users\you\Downloads\nimbus_triage_test_scenarios.pdf
Length of page : 5
Length of chunk : 10
Enter your ticket (ticket-id b958fe36-...) : Hi, my annual plan renewed yesterday for $144 and I do not want it. Please refund me.

--- DRAFT ---
Subject: Refund request for your annual plan renewal
...

approve/edit/reject the draft? approve
```

- `approve` → the draft becomes the final answer, the run ends.
- `edit` → type what should change, the drafter regenerates (max 3 edits).
- `reject` → run ends with *"User rejected the request."*.
- `exit` / `quit` / `q` at the ticket prompt quits the CLI.

---

## Screenshots

### Retrieval smoke test — `test.py`

Hybrid search (MMR + BM25 + Cohere re-ranking) returning scored chunks from the policy PDF:

![Retrieval smoke test output](docs/retrieval_test.png)

### End-to-end run — draft approved

Classify → query → retrieve → draft → grader passes → human approves:

![Ticket approved output](docs/ticket_approved.png)

### End-to-end run — draft edited with feedback, then approved

The human rejects the first draft with feedback and the graph loops back to the drafter:

![Ticket edited output](docs/ticket_edited.png)

---

## Configuration

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | yes | Groq LLM (`ChatGroq`) |
| `COHERE_API_KEY` | yes | Cohere re-ranking |
| `DATABASE_URL` | yes | PostgreSQL checkpoint store |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGSMITH_ENDPOINT` | no | Tracing in LangSmith |

Model and retriever settings live in `backend.py` (`llm = ChatGroq(model="qwen/qwen3.8-27b")`) and `rag.py` (embeddings, chunk size, ensemble weights, re-ranker).

---

## Notes and limitations

- `rag.py` asks for the policy PDF path **at import time** (`input()`), so both `backend.py` and `test.py` prompt for it before anything else runs.
- The Chroma store in `./chroma_pdr` is append-only across runs; delete the folder to rebuild the index from scratch.
- Reply length is capped at ~150 words and every fact must come from the retrieved context — when the context is silent the agent says a team member will follow up.
- Tickets are untrusted input: all prompts instruct the model to classify/draft *about* the ticket, never to follow instructions written inside it.
- The classifier, grader and drafter use structured output (`pydantic` schemas), so the graph never parses free-form JSON.

---

## License

[MIT](LICENSE) © 2026 Yogeshwaran D
