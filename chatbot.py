"""
chatbot.py
----------
Script 3 — CLI Chatbot with Top-K Aggregation.

Key features:
  * No LLM at runtime: zero hallucinations, minimal latency, near-zero cost.
  * Score aggregation by `answer_id`: if multiple question variants for the
    same answer match the query, their scores are summed, making the winner
    much more robust than relying on the single top-1 hit.
  * OOD threshold configurable via --threshold.
  * Automatic logging of "missed" queries to missed_queries.log.

Usage:
    python chatbot.py
    python chatbot.py --threshold 0.75
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import litellm
from litellm import embedding

from qdrant_client import QdrantClient

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
def load_answers_dict(path: str) -> dict[str, dict]:
    """Load data.json and build an id -> block dict for O(1) lookup."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] {path} not found. Run `prepare_data.py` and `ingest_to_qdrant.py` first.")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def embed_query(query: str) -> list[float]:
    """
    Single embedding for the user query.

    Uses task_type=RETRIEVAL_QUERY (asymmetric with the RETRIEVAL_DOCUMENT
    task type used at ingestion time): this is the recommended setup for
    Gemini embeddings and gives noticeably better retrieval quality.
    """
    resp = embedding(
        model=config.EMBEDDING_MODEL,
        input=[query],
        num_retries=config.LITELLM_NUM_RETRIES,
        task_type=config.EMBED_TASK_TYPE_QUERY,
        output_dimensionality=config.VECTOR_SIZE,
    )
    return resp["data"][0]["embedding"]


def log_missed_query(query: str, top_score: float) -> None:
    """Append-only log of below-threshold queries."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{ts}\tscore={top_score:.4f}\tquery={query}\n"
    try:
        with open(config.MISSED_QUERIES_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        # Logging must never crash the chatbot loop.
        print(f"[WARN] Could not write missed-query log: {e}")


def aggregate_scores(search_results) -> dict[str, dict]:
    """
    Aggregate scores of the Top-K Qdrant results by `answer_id`.

    Returns a dict:
      { answer_id: {
            "score": float (sum),
            "hits": int (number of matching variants),
            "best_question": str (the variant with the highest single score),
            "best_single_score": float
        }, ... }
    """
    agg: dict[str, dict] = defaultdict(lambda: {
        "score": 0.0,
        "hits": 0,
        "best_question": "",
        "best_single_score": -1.0,
    })

    for r in search_results:
        payload = r.payload or {}
        aid = payload.get("answer_id")
        if not aid:
            continue
        bucket = agg[aid]
        score = float(r.score)
        bucket["score"] += score
        bucket["hits"] += 1
        if score > bucket["best_single_score"]:
            bucket["best_single_score"] = score
            bucket["best_question"] = payload.get("question_text", "")

    return agg


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def answer_query(
    query: str,
    client: QdrantClient,
    answers_dict: dict[str, dict],
    threshold: float,
    debug: bool = False,
) -> None:
    """Run retrieval + aggregation and print up to the top 3 best answers."""
    try:
        qvec = embed_query(query)
    except Exception as e:
        print(f"\n[ERROR] Embedding failed: {e}")
        return

    try:
        results = client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=qvec,
            limit=config.TOP_K_RETRIEVAL,
            with_payload=True,
        ).points
    except Exception as e:
        print(f"\n[ERROR] Qdrant query failed: {e}")
        return

    if not results:
        print("\nI couldn't find any relevant information.")
        log_missed_query(query, 0.0)
        return

    agg = aggregate_scores(results)
    
    # --- DEBUG: Top-K matches ---
    if debug:
        print("\n--- DEBUG TOP-K ---")
        for r in results:
            payload = r.payload or {}
            print(f"  score={r.score:.4f}  aid={payload.get('answer_id')}  q={payload.get('question_text')}")

    if not agg:
        print("\nI couldn't find any relevant information.")
        log_missed_query(query, 0.0)
        return

    # --- LOGIC: Filter and sort candidates ---
    candidates = []
    for aid, b in agg.items():
        passes_agg = b["score"] >= threshold
        passes_single = b["best_single_score"] >= config.SINGLE_HIT_THRESHOLD
        min_quality = b["best_single_score"] > 0.68
        
        if (passes_agg or passes_single) and min_quality:
            candidates.append((aid, b))

    if not candidates:
        print("\nI couldn't find any relevant information on this topic in my knowledge base.")
        top_agg_score = max(b["score"] for b in agg.values())
        log_missed_query(query, top_agg_score)
        return

    candidates.sort(key=lambda x: x[1]["score"], reverse=True)
    top_candidates = candidates[:3]

    print(f"\n[INFO] Found {len(candidates)} relevant answers, showing top {len(top_candidates)}:")

    for i, (aid, stats) in enumerate(top_candidates, 1):
        block = answers_dict.get(aid)
        if not block:
            continue
            
        print("\n" + "─" * 70)
        print(f"--- Answer {i} (Pertinence: {stats['score']:.2f}) ---")
        print(block["answer"])
        print("─" * 70)
        print(f"Source: {block.get('source_file', 'n/a')}")

    # --- DEBUG: Aggregation table ---
    if debug:
        print(f"\n[debug] Displayed {len(top_candidates)} answers out of {len(candidates)} found.")
        print("--- DEBUG AGG (All candidates) ---")
        for aid, b in candidates:
            print(f"  aid={aid}  agg={b['score']:.4f}  hits={b['hits']}  best_single={b['best_single_score']:.4f}")


# -----------------------------------------------------------------------------
# CLI Loop
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CLI chatbot, Q-Q matching on local Qdrant.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DEFAULT_THRESHOLD,
        help=f"Minimum aggregated-score threshold (default: {config.DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show internal details (Top-K and aggregation table).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RAGless — Script 3: chatbot.py")
    print(f"  Embedding model : {config.EMBEDDING_MODEL}")
    print(f"  Qdrant path     : {config.QDRANT_PATH}")
    print(f"  Collection      : {config.COLLECTION_NAME}")
    print(f"  Top-K           : {config.TOP_K_RETRIEVAL}")
    print(f"  Threshold (agg) : {args.threshold}")
    print(f"  Threshold (sgl) : {config.SINGLE_HIT_THRESHOLD}")
    print(f"  Debug           : {args.debug} ( Use --debug flag to see scores and aggregation details. )")
    print("=" * 70)
    print("Type your question. Type 'exit' or 'quit' to leave.\n")
    
    answers_dict = load_answers_dict(config.DATA_JSON)
    client = QdrantClient(path=config.QDRANT_PATH)

    # Sanity check: does the collection exist?
    try:
        info = client.get_collection(config.COLLECTION_NAME)
        print(f"[INFO] Collection ok — {info.points_count} points loaded.\n")
    except Exception:
        print(f"[ERROR] Collection '{config.COLLECTION_NAME}' not found. "
              f"Run `python ingest_to_qdrant.py` first.")
        try:
            client.close()
        except Exception:
            pass
        sys.exit(1)

    all_questions = [q for block in answers_dict.values() for q in block.get("questions", [])]
    if all_questions:
        sample_size = min(10, len(all_questions))
        examples = random.sample(all_questions, sample_size)
        print("💡 Here are some example questions you can ask me::")
        for q in examples:
            print(f"  • {q}")
        print()
                
    try:
        while True:
            try:
                query = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit", ":q"}:
                print("Goodbye!")
                break

            answer_query(
                query=query,
                client=client,
                answers_dict=answers_dict,
                threshold=args.threshold,
                debug=args.debug,
            )
            print()
    finally:
        # Qdrant local mode keeps a lockfile: explicit close on exit.
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
