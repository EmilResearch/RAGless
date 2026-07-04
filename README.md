# RAGless — Deterministic Retrieval-Only Q&A System

**RAGless** is a retrieval-only question-answering system with **zero LLM calls at runtime**.
Source documents are converted into self-contained informational blocks, indexed into a local vector database (Qdrant), and queried via asymmetric Gemini embeddings.

> **Zero hallucinations at runtime. Minimal latency. Near-zero cost per query.**

![How does RAGless work](./ragless.png)

[See it in action](https://youtu.be/AQnVMuAVPBA)

---

## Overview

The project consists of three independent scripts:

| Script | Purpose |
|--------|---------|
| `prepare_data.py` | Extracts Q&A blocks from source documents using Gemini in JSON mode |
| `ingest_to_qdrant.py` | Generates embeddings and populates the local Qdrant vector database |
| `chatbot.py` | CLI chatbot that retrieves the most relevant answers for a user query |

---

## Key Features

- **No LLM at runtime** — The chatbot relies purely on vector retrieval. No expensive API calls during user interaction.
- **Robust Q-Q matching via `answer_id` aggregation** — If multiple question variants for the same answer match the query, their scores are summed. This makes the result far more stable than classic "top-1" retrieval.
- **Asymmetric Gemini embeddings** — `RETRIEVAL_DOCUMENT` at ingestion time and `RETRIEVAL_QUERY` at retrieval time, as recommended by Google.
- **Embedded Qdrant** — No Docker server, no cloud service. Data is stored locally on disk (`./qdrant_data`).
- **Smart chunking** — Documents are chunked only if they exceed a token threshold, measured with the model's real tokenizer.
- **Optional Judge verification** — `prepare_data.py` can enable a second LLM pass to discard blocks not supported by the source text (`--judge`).
- **Missed query logging** — Below-threshold queries are automatically logged to `missed_queries.log` for later analysis.
- **Guaranteed idempotency** — Every run of `ingest_to_qdrant.py` recreates the collection from scratch, preventing hidden duplicates.

---

## Requirements

- Python 3.10+
- [Gemini API Key](https://aistudio.google.com/app/apikey) (free tier with generous limits)
- Python dependencies (see Installation section)

---

## Installation

1. **Clone or download the repository** and navigate to the project folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install litellm qdrant-client pypdf python-dotenv tqdm
   ```

4. **Configure your API key**:
   ```bash
   cp .env.example .env
   # Edit .env and insert your GEMINI_API_KEY
   ```

---

## Project Structure

```
.
├── source/                    # Folder with source documents (.pdf, .txt, .md)
├── config.py                  # Centralized configuration (models, thresholds, paths)
├── prepare_data.py            # Script 1: Q&A block extraction
├── ingest_to_qdrant.py        # Script 2: embedding and indexing
├── chatbot.py                 # Script 3: CLI chatbot
├── data.json                  # Output of prepare_data.py (validated blocks)
├── qdrant_data/               # Local vector database (auto-created)
├── failed_chunks/             # Chunks that failed to produce valid JSON (debug)
└── missed_queries.log         # Log of below-threshold queries
```

---

## Usage

### 1. Prepare the data (`prepare_data.py`)

Place your documents in the `source/` folder (supports `.pdf`, `.txt`, `.md`), then run:

```bash
python prepare_data.py
```

With optional Judge verification (slower but more accurate):
```bash
python prepare_data.py --judge
```

**What it does:**
- Reads each file and counts tokens.
- If the document is short (≤ 10,000 tokens), sends it whole to the LLM; otherwise splits it into chunks.
- Extracts JSON blocks with `answer`, `questions`, `category`, `source_quote`.
- Validates blocks and saves them to `data.json`.

---

### 2. Index into the vector database (`ingest_to_qdrant.py`)

```bash
python ingest_to_qdrant.py
```

**What it does:**
- Loads `data.json`.
- "Explodes" each block into as many rows as its question variants.
- Generates embeddings in batches via LiteLLM + Gemini.
- Recreates the Qdrant collection and inserts vectors with deterministic UUID5s.

---

### 3. Launch the chatbot (`chatbot.py`)

```bash
python chatbot.py
```

Available options:
```bash
python chatbot.py --threshold 0.75   # Change the minimum aggregated score threshold
python chatbot.py --debug            # Show internal scores and aggregation table
```

**Interaction:**
```
You> How does check-in work?
[INFO] Found 2 relevant answers, showing top 1:
──────────────────────────────────────────────────────────────────────
--- Answer 1 (Pertinence: 1.85) ---
Check-in is available from 3:00 PM to 8:00 PM. If you arrive after 8:00 PM,
please contact reception in advance...
──────────────────────────────────────────────────────────────────────
Source: source/regulations.txt
```

Type `exit`, `quit`, or `:q` to leave.

---

## How Retrieval Works

The core of the chatbot is **aggregation by `answer_id`**:

1. The user query is embedded with `task_type=RETRIEVAL_QUERY`.
2. Qdrant returns the `TOP_K_RETRIEVAL` most similar points (questions).
3. Scores of points pointing to the same `answer_id` are **summed**.
4. A candidate is shown only if:
   - the aggregated score exceeds `DEFAULT_THRESHOLD` **OR**
   - the best single hit exceeds `SINGLE_HIT_THRESHOLD`
   - **AND** the best single hit is > 0.68 (minimum quality)

This mechanism makes the system robust: even if no single question variant is a perfect match, the sum of multiple weak matches on the same answer can make it emerge correctly.

---

## RAGless vs. Classic Generative RAG

### Philosophy

Classic RAG generates answers at runtime by retrieving context and prompting an LLM to synthesize a response. **RAGless eliminates the generative step entirely.** Answers are pre-generated during ingestion and retrieved verbatim at runtime.

### Advantages of RAGless

| Advantage | Explanation |
|-----------|-------------|
| **Much simpler pipeline** | No prompt engineering for answer generation, no context window management, no output parsing. Just embed, search, return. |
| **Q-Q matching is far more reliable** | Matching query-to-question (Q-Q) is semantically easier and more robust than query-to-document-chunk (Q-D) or query-to-answer (Q-A). Multiple question variants per answer provide redundancy. |
| **Determinism** | Same question, same answer, always. Behavior is reproducible and testable. |
| **Zero hallucinations at runtime** | No LLM generates answers at query time. Returned text is pre-generated and immutable. |
| **Zero cost per query** | After ingestion, there are no API calls. Retrieval is purely local computation. |
| **Low latency** | Query embedding + vector search. Milliseconds, not seconds. |
| **Verifiable answers** | Every block has `source_quote` and `source_file`. Complete audit trail. |
| **Reproducible bugs** | If an answer is wrong, it is 100% wrong. Easy to find and fix. |
| **Finite output surface** | The number of possible answers is known (`data.json`). Exhaustively testable. |
| **Runs on modest hardware** | Embedded Qdrant, no LLM in memory. Works on CPU with a few GB of RAM. |
| **Total privacy** | No user data leaves the machine after ingestion. |
| **No dependency drift** | The embedding model can change, but the answers do not. You are not tied to an LLM provider's availability or pricing. |

### Trade-offs and Limitations

| Limitation | Explanation |
|------------|-------------|
| **No real-time flexibility** | Cannot synthesize novel answers, combine information across blocks, or adapt tone dynamically. What you ingest is what you get. |
| **Higher ingestion cost** | Using an LLM to generate Q&A blocks costs more than simple embedding. See cost comparison below. |
| **Coverage bounded by ingestion** | If a topic was not extracted during `prepare_data.py`, the system cannot answer it. No "reasoning" around gaps. |
| **Maintenance requires re-ingestion** | Updating answers requires re-running the full pipeline, not just editing a prompt. |

### Cost & Performance Comparison

| Metric | Classic Generative RAG | Q-Q System (This Project) | Difference / Advantage |
|--------|------------------------|---------------------------|------------------------|
| **Ingestion Cost (One-time)** | ~$0.01 *(embedding 100,000 tokens only)* | ~$1.50 *(LLM generates Q&A from 100,000 tokens)* | **+$1.49** (Initial disadvantage, but negligible cost) |
| **Runtime API Cost (Monthly)** | ~$157.50 *(1,000 queries/day, ~1,000 context tokens + 150 output tokens)* | **$0.00** *(only embedding 1,000 short queries, < $0.02/month)* | **~$157.50/month saved** (Zero-cost scalability) |
| **Latency (per query)** | 2.5 – 4 seconds *(LLM text generation)* | **~0.15 seconds** *(pure vector search)* | **>15x faster** (Instant response) |
| **Hallucination Rate (Runtime)** | Low, but always > 0% | **0%** | **Risk eliminated** (Static, deterministic output) |

### The Hallucination Trade-off

RAGless eliminates hallucinations **at runtime**, where they are most dangerous because they are uncontrollable. The risk during ingestion still exists, but it is mitigated — and crucially — **it happens offline, in a controlled environment, with the possibility of human review** before the knowledge base goes to production.

RAGless shifts the hallucination risk from runtime to ingestion. Generation happens during `prepare_data.py`, which is why the optional `--judge` pass and deterministic UUIDs for idempotency were added.

> **The trade-off is:** you give up real-time flexibility in exchange for offline verifiability. For high-risk domains, I prefer hallucinations I can catch in a log over ones I cannot predict.

---

## Advanced Configuration

All tunable constants are in `config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `LLM_MODEL` | Gemini model for extraction and judge | `gemini/gemini-2.5-flash` |
| `EMBEDDING_MODEL` | Embedding model | `gemini/gemini-embedding-001` |
| `VECTOR_SIZE` | Vector dimension (Matryoshka) | `3072` |
| `MAX_TOKENS_DOC` | Threshold for sending whole document | `10_000` |
| `CHUNK_SIZE` / `OVERLAP` | Chunk size and overlap | `8_000` / `500` |
| `TOP_K_RETRIEVAL` | Candidates retrieved from Qdrant | `10` |
| `DEFAULT_THRESHOLD` | Minimum aggregated score threshold | `1.35` |
| `SINGLE_HIT_THRESHOLD` | Fallback threshold on best single hit | `0.75` |
| `EMBEDDING_BATCH_SIZE` | Questions per embedding API call | `100` |
| `QDRANT_UPSERT_BATCH` | Points per upsert batch | `256` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `GEMINI_API_KEY not found` | Create `.env` file with `GEMINI_API_KEY=...` |
| `Collection not found` | Run `python ingest_to_qdrant.py` first |
| Malformed JSON in chunks | Check the `failed_chunks/` folder for raw text |
| Empty LLM response | Possible Gemini safety block; try reducing or modifying source text |
| Qdrant lockfile | Client closes automatically; in case of crash, manually remove the lock in `qdrant_data/` |

---

## Technical Notes

- **LiteLLM** is used as a unified proxy to call Gemini for both completions and embeddings.
- **Qdrant Client** in `path=` mode stores everything in local files: no server process is required.
- Qdrant point IDs are deterministic UUID5s (`uuid5(NAMESPACE_DNS, answer_id:question_text)`), so re-running ingestion does not create logical duplicates.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

See [LICENSE](LICENSE) for details.
