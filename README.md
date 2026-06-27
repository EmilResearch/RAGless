# SemanticFAQ

SemanticFAQ is a lightweight semantic Question-to-Question (Q-Q) retrieval system designed for FAQ search.

Unlike traditional RAG systems, **SemanticFAQ does not use an LLM during inference**. Responses are retrieved directly from a curated knowledge base using semantic similarity, making the system:

* Fast
* Deterministic
* Hallucination-free
* Low-cost (no runtime LLM calls)
* Easy to deploy locally

LLMs are only used during the **ingestion phase** to transform unstructured documents into high-quality Question & Answer pairs.

---

# Features

* Semantic FAQ retrieval using embeddings
* No LLM required during user queries
* Automatic FAQ generation from PDF, TXT and Markdown documents
* Multiple semantic question variants for every answer
* Local Qdrant vector database
* JSON document store
* Configurable similarity threshold
* Optional LLM-as-a-Judge validation
* Failed JSON recovery
* Logging of unanswered user queries
* Batch embedding generation for fast ingestion

---

# Project Architecture

```
Raw Documents
      │
      ▼
prepare_data.py
      │
      ▼
Generated FAQ Blocks (data.json)
      │
      ▼
ingest_to_qdrant.py
      │
      ▼
Qdrant Vector Database
      │
      ▼
chatbot.py
      │
      ▼
Semantic Retrieval
```

---

# Project Structure

```
SemanticFAQ/
│
├── source/
│   ├── manual.pdf
│   ├── faq.txt
│   └── notes.md
│
├── failed_chunks/
│
├── config.py
├── prepare_data.py
├── ingest_to_qdrant.py
├── chatbot.py
├── data.json
├── missed_queries.log
├── requirements.txt
├── .env
└── README.md
```

---

# How It Works

SemanticFAQ consists of two completely independent phases:

1. Ingestion
2. Retrieval

The expensive AI processing happens only once during ingestion.

Runtime retrieval is extremely lightweight.

---

# Ingestion Flow

The ingestion pipeline converts raw documents into a searchable semantic knowledge base.

```
Documents
(PDF / TXT / MD)

        │

        ▼

Extract Text

        │

        ▼

Token Counting

        │

        ▼

Chunking
(if necessary)

        │

        ▼

Gemini
Q&A Generation

        │

        ▼

Generate
Question Variants

        │

        ▼

(Optional)
LLM-as-a-Judge

        │

        ▼

Save FAQ Blocks
(data.json)

        │

        ▼

Generate Embeddings

        │

        ▼

Store Embeddings
inside Qdrant
```

Each generated FAQ block contains:

* Unique ID
* Answer
* Multiple semantic question variants
* Category
* Source file
* Source quote

Every question variant receives its own embedding.

---

# Retrieval Flow

No LLM is involved during retrieval.

```
User Question

      │

      ▼

Generate Embedding

      │

      ▼

Search Top-K
Similar Questions

      │

      ▼

Aggregate Scores
by Answer ID

      │

      ▼

Best Matching Answer

      │

      ▼

Threshold Check

      │

      ├──── Below threshold
      │         │
      │         ▼
      │   Log missed query
      │
      ▼

Return Answer
+
Source File
```

Instead of selecting only the single nearest question, SemanticFAQ aggregates similarity scores belonging to the same answer.

This makes retrieval considerably more robust when several different question formulations refer to the same answer.

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/SemanticFAQ.git

cd SemanticFAQ
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

# Prepare Your Documents

Copy your documents into the `source/` directory.

Supported formats:

* PDF
* TXT
* Markdown

Example

```
source/

hotel_manual.pdf

faq.md

notes.txt
```

---

# Step 1 — Generate the Knowledge Base

Run:

```bash
python prepare_data.py
```

If you also want AI validation:

```bash
python prepare_data.py --judge
```

This script will:

* Read every document
* Split large files into chunks
* Generate FAQ blocks using Gemini
* Optionally validate every answer
* Save everything into `data.json`

---

# Step 2 — Build the Vector Database

Run:

```bash
python ingest_to_qdrant.py
```

This script will:

* Read `data.json`
* Generate embeddings
* Create a local Qdrant collection
* Store every question embedding

---

# Step 3 — Start the Chatbot

Run:

```bash
python chatbot.py
```

Example

```
You:
How do I connect to the hotel Wi-Fi?

Assistant:
The Wi-Fi password is available at the reception...

Source:
hotel_manual.pdf
```

---

# Custom Threshold

You can adjust the similarity threshold.

Example

```bash
python chatbot.py --threshold 0.75
```

Higher values

* fewer false positives
* more unanswered questions

Lower values

* higher recall
* increased risk of incorrect matches

---

# Configuration

All configuration is centralized inside:

```
config.py
```

Examples include:

* embedding model
* LLM model
* chunk size
* overlap
* retrieval Top-K
* similarity threshold
* Qdrant collection name
* vector size

No magic numbers are scattered throughout the code.

---

# Output Files

## data.json

Contains the generated FAQ knowledge base.

Example

```json
{
    "id": "...",
    "answer": "...",
    "questions": [
        "...",
        "...",
        "..."
    ],
    "category": "...",
    "source_file": "...",
    "source_quote": "..."
}
```

---

## failed_chunks/

If Gemini returns malformed JSON, the original text chunk is automatically saved here for manual inspection.

---

## missed_queries.log

Whenever no answer passes the similarity threshold, the user query is logged together with:

* timestamp
* similarity score
* original query

This makes it easy to identify missing knowledge and continuously improve the FAQ database.

---

# Why Not RAG?

Traditional Retrieval-Augmented Generation (RAG) pipelines typically work like this:

```
Retrieve Documents

        │

        ▼

Send Context
to an LLM

        │

        ▼

Generate Answer
```

SemanticFAQ follows a different philosophy:

```
Retrieve Questions

        │

        ▼

Return Existing Answer
```

Advantages:

* No hallucinations
* No prompt engineering
* No runtime LLM cost
* Predictable answers
* Very low latency

This architecture is ideal for:

* FAQ systems
* Internal documentation
* Product manuals
* Company knowledge bases
* Customer support
* Hotel assistants
* Appliance manuals
* Policy documents

---

# Future Improvements

Possible future enhancements include:

* Cross-encoder reranking
* Automatic evaluation dataset
* Incremental ingestion
* Metadata filtering
* REST API
* Web interface
* Telegram integration
* Multi-language support
* Confidence estimation
* SQLite document store
* Docker deployment

---

# License

This project is released under the MIT License.

