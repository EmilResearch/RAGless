"""
ingest_to_qdrant.py
-------------------
Script 2 — Batch embedding and Vector DB population (local Qdrant).

Pipeline:
  1. Reads `data.json`.
  2. "Explodes" every question of every block while keeping a reference
     to the answer_id (the block id) and the category.
  3. Generates embeddings in batches (EMBEDDING_BATCH_SIZE per call) using
     the RETRIEVAL_DOCUMENT task type (asymmetric embeddings).
  4. (Re)creates the Qdrant collection and upserts every vector.

Idempotency: on every run the collection is recreated from scratch,
so hidden duplicates never accumulate.

Usage:
    python ingest_to_qdrant.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import litellm
from litellm import embedding

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("[ERROR] GEMINI_API_KEY not found. Copy .env.example to .env and set the key.")
    sys.exit(1)

litellm.suppress_debug_info = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_data(path: str) -> list[dict]:
    """Load data.json and do minimal validation."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] {path} not found. Run `python prepare_data.py` first.")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print(f"[ERROR] {path} is empty or invalid.")
        sys.exit(1)

    return data


def explode_questions(blocks: list[dict]) -> list[dict]:
    """
    Expand every block into its question variants, keeping a back-reference
    to the parent block (answer_id).
    """
    rows = []
    for block in blocks:
        for q in block.get("questions", []):
            if not isinstance(q, str):
                continue
            q = q.strip()
            if not q:
                continue
            rows.append({
                "answer_id": block["id"],
                "category": block.get("category", ""),
                "question_text": q,
                "source_file": block.get("source_file", ""),
            })
    return rows


def embed_in_batches(texts: list[str], batch_size: int) -> list[list[float]]:
    """
    Generate embeddings in batches using LiteLLM.
    The order of the returned vectors matches the order of the input texts.

    We use the RETRIEVAL_DOCUMENT task type for ingestion, and at query time
    the chatbot uses RETRIEVAL_QUERY. This asymmetric setup is the recommended
    way to use Gemini embeddings and improves retrieval quality noticeably.
    """
    vectors: list[list[float]] = []
    n = len(texts)

    for start in tqdm(range(0, n, batch_size), desc="Embedding batches", unit="batch"):
        batch = texts[start:start + batch_size]
        resp = embedding(
            model=config.EMBEDDING_MODEL,
            input=batch,
            num_retries=config.LITELLM_NUM_RETRIES,
            # Gemini-specific kwargs forwarded by LiteLLM.
            task_type=config.EMBED_TASK_TYPE_DOCUMENT,
            output_dimensionality=config.VECTOR_SIZE,
        )
        # LiteLLM normalizes the output OpenAI-style: resp["data"] is a list of
        # dicts with key "embedding". The order is guaranteed to match the input.
        batch_vecs = [item["embedding"] for item in resp["data"]]
        if len(batch_vecs) != len(batch):
            raise RuntimeError(
                f"Embedding batch length mismatch: expected {len(batch)}, got {len(batch_vecs)}"
            )
        vectors.extend(batch_vecs)

    return vectors


def create_collection(client: QdrantClient) -> None:
    """Recreate the collection from scratch — idempotency guaranteed."""
    # Use the new methods to avoid the DeprecationWarning.
    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)

    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=config.VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"[INFO] Collection '{config.COLLECTION_NAME}' recreated "
          f"(size={config.VECTOR_SIZE}, distance=COSINE).")


def upsert_points(client: QdrantClient, rows: list[dict], vectors: list[list[float]]) -> None:
    """Insert every question (with its embedding) as a separate point."""
    if len(rows) != len(vectors):
        raise RuntimeError(
            f"Row/vector count mismatch: {len(rows)} rows vs {len(vectors)} vectors"
        )

    points = []
    for row, vec in zip(rows, vectors):
        points.append(
            qmodels.PointStruct(
                # Deterministic UUID5: re-running ingest produces the same id
                # for the same (answer_id, question_text) pair.
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{row['answer_id']}:{row['question_text']}")),
                vector=vec,
                payload={
                    "answer_id": row["answer_id"],
                    "category": row["category"],
                    "question_text": row["question_text"],
                    "source_file": row["source_file"],
                },
            )
        )

    # Upsert in batches to avoid saturating memory on large collections.
    batch = config.QDRANT_UPSERT_BATCH
    for start in tqdm(range(0, len(points), batch), desc="Upserting", unit="batch"):
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points[start:start + batch],
            wait=True,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("RAGless — Script 2: ingest_to_qdrant.py")
    print(f"  Embedding model : {config.EMBEDDING_MODEL}")
    print(f"  Vector size     : {config.VECTOR_SIZE}")
    print(f"  Qdrant path     : {config.QDRANT_PATH}")
    print(f"  Collection      : {config.COLLECTION_NAME}")
    print(f"  Batch size      : {config.EMBEDDING_BATCH_SIZE}")
    print("=" * 70)

    blocks = load_data(config.DATA_JSON)
    print(f"\n[INFO] Blocks loaded: {len(blocks)}")

    rows = explode_questions(blocks)
    if not rows:
        print("[ERROR] No questions found in the blocks.")
        sys.exit(1)
    print(f"[INFO] Total questions to embed: {len(rows)}")

    texts = [r["question_text"] for r in rows]
    print(f"[INFO] Starting embedding in batches of {config.EMBEDDING_BATCH_SIZE}...")
    vectors = embed_in_batches(texts, batch_size=config.EMBEDDING_BATCH_SIZE)
    print(f"[INFO] Embeddings generated: {len(vectors)}")

    # Initialize Qdrant in local mode (on-disk path, no server required).
    client = QdrantClient(path=config.QDRANT_PATH)
    try:
        create_collection(client)
        upsert_points(client, rows, vectors)

        # Verification
        info = client.get_collection(config.COLLECTION_NAME)
        print("\n" + "=" * 70)
        print(f"DONE. Points in collection: {info.points_count}")
        print(f"Qdrant data: {config.QDRANT_PATH}")
        print("=" * 70)
    finally:
        # Qdrant local mode keeps a lockfile: explicit close.
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
