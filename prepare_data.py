"""
prepare_data.py
---------------
Script 1 — Data generation for the FAQ Retrieval system.

Pipeline:
  1. Recursively scans the `source/` folder (pdf, txt, md).
  2. Counts tokens with `litellm.token_counter` (NO tiktoken).
  3. If the document has <= MAX_TOKENS_DOC tokens -> send it whole;
     otherwise chunk it with CHUNK_SIZE / OVERLAP.
  4. Sends each chunk to Gemini in JSON mode using the Q&A extraction prompt.
  5. (Optional --judge) Verifies every block with a second LLM-as-a-Judge pass.
  6. Saves the array of blocks to `data.json`.

Usage:
    python prepare_data.py
    python prepare_data.py --judge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

import litellm
from litellm import completion, token_counter

# pypdf for PDF parsing
from pypdf import PdfReader

import config

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("[ERROR] GEMINI_API_KEY not found. Copy .env.example to .env and set the key.")
    sys.exit(1)

# Silence LiteLLM verbose logs (we keep only our structured prints).
litellm.suppress_debug_info = True


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an expert Information Retrieval and Data Ingestion system for semantic search engines.

Your goal is to convert the provided documents into a set of independent informational blocks.

Strictly follow these guidelines:
1. FULL COVERAGE: Do not omit any detail, technical data, procedure, or exception present in the text. All informational value must be extracted.
2. BLOCK STRUCTURE: Each block must contain:
   - "answer": A comprehensive, clear, and self-contained text explaining a specific concept or procedure.
   - "questions": A list of 3-5 question variants that are answered *exactly* and *only* by that text (vary style: formal, colloquial, keyword-focused).
   - "category": A short category for the block (e.g., "Check-in", "Appliances", "Payments").
   - "source_quote": The EXACT sentence or paragraph from which you extracted the answer (copied literally).
3. ISOLATION: Answers must not require reading the rest of the document.
4. DO NOT HALLUCINATE: Extract only information explicitly written. If the text is vague, ignore it.

Generate the output EXCLUSIVELY in JSON format (an object with key "blocks" containing the array of blocks):
{
  "blocks": [
    {
      "answer": "Full text...",
      "questions": ["Question 1?", "Question 2?", "Question 3?"],
      "category": "Category Name",
      "source_quote": "Literal text from the document..."
    }
  ]
}

Here is the content to process:
"""

JUDGE_PROMPT_TEMPLATE = """You are a cross-verification system. You are provided with a document excerpt (Source Quote) and a generated answer (Answer).
Verify whether the Answer is 100% supported by the Source Quote, with no additions or external inferences.
Respond EXCLUSIVELY in JSON format:
{{
  "verdict": "YES" or "NO",
  "reasoning": "Brief explanation"
}}

Source Quote:
{source_quote}

Answer:
{answer}
"""


# -----------------------------------------------------------------------------
# File reading
# -----------------------------------------------------------------------------
def read_file(path: Path) -> str:
    """Read the textual content of a supported file (.pdf, .txt, .md)."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            print(f"  [WARN] Could not open PDF {path.name}: {e}")
            return ""
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:
                print(f"  [WARN] Page extraction error on {path.name}: {e}")
        return "\n".join(pages)

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")

    return ""


def discover_files(source_dir: str) -> list[Path]:
    """Recursively walk the source folder and return the supported file paths."""
    root = Path(source_dir)
    if not root.exists():
        print(f"[ERROR] Source folder not found: {source_dir}")
        sys.exit(1)

    supported = {".pdf", ".txt", ".md"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in supported]
    return files


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Count tokens using the LLM tokenizer (via LiteLLM)."""
    return token_counter(model=config.LLM_MODEL, text=text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Approximate token-based chunking via the actual token/char ratio.

    LiteLLM does not expose a public tokenizer for Gemini, so we compute the
    character length equivalent to CHUNK_SIZE tokens using the actual char/token
    ratio measured on the input text. This works well for natural-language text.
    """
    total_tokens = count_tokens(text)
    if total_tokens <= chunk_size:
        return [text]

    # char/token ratio of the *actual* text (more accurate than a fixed estimate).
    char_per_token = max(1.0, len(text) / total_tokens)
    chunk_chars = int(chunk_size * char_per_token)
    overlap_chars = int(overlap * char_per_token)
    step = max(1, chunk_chars - overlap_chars)

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step

    return chunks


# -----------------------------------------------------------------------------
# LLM calls
# -----------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_json_fences(raw: str) -> str:
    """
    Remove possible Markdown code fences around a JSON payload.

    Some Gemini responses still include ```json ... ``` even when the API is
    called in JSON mode. Stripping them defensively avoids JSONDecodeError.
    """
    return _JSON_FENCE_RE.sub("", raw).strip()


def extract_blocks_from_chunk(chunk_content: str) -> dict[str, Any] | None:
    """
    Send a chunk to the LLM in JSON mode and return the parsed dict.
    Returns None if the call fails or the JSON is invalid.

    NOTE: the parameter is named `chunk_content` (not `chunk_text`) to avoid
    shadowing the module-level `chunk_text` function.
    """
    try:
        response = completion(
            model=config.LLM_MODEL,
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT + chunk_content}
            ],
            response_format={"type": "json_object"},
            num_retries=config.LITELLM_NUM_RETRIES,
        )
        raw = response["choices"][0]["message"]["content"]
        if not raw:
            # Gemini can return an empty content when a safety filter triggers.
            print("  [WARN] Empty LLM response (possible safety block).")
            return None
        return json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as e:
        print(f"  [WARN] Malformed JSON: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] LLM error: {e}")
        return None


def judge_block(source_quote: str, answer: str) -> bool:
    """
    Verify a block with a second LLM pass. Returns True if SUPPORTED ("SI"),
    False if not supported. On call failure we are conservative and keep the
    block (returning True), so transient API issues don't silently drop content.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(source_quote=source_quote, answer=answer)
    try:
        response = completion(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            num_retries=config.LITELLM_NUM_RETRIES,
        )
        raw = response["choices"][0]["message"]["content"]
        if not raw:
            print("  [WARN] Judge returned empty content (keeping block).")
            return True
        verdict = json.loads(_strip_json_fences(raw))
        return str(verdict.get("verdict", "")).strip().upper() == "YES"
    except Exception as e:
        print(f"  [WARN] Judge failed (keeping block): {e}")
        return True

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
REQUIRED_KEYS = {"answer", "questions", "category", "source_quote"}


def validate_block(block: Any) -> bool:
    """Check that a block has every required key and sane types."""
    if not isinstance(block, dict):
        return False
    if not REQUIRED_KEYS.issubset(block.keys()):
        return False
    if not isinstance(block["questions"], list) or len(block["questions"]) == 0:
        return False
    if not all(isinstance(q, str) and q.strip() for q in block["questions"]):
        return False
    if not isinstance(block["answer"], str) or not block["answer"].strip():
        return False
    if not isinstance(block["source_quote"], str) or not block["source_quote"].strip():
        return False
    if not isinstance(block.get("category", ""), str):
        return False
    return True


def save_failed_chunk(filename: str, chunk_idx: int, chunk_content: str) -> None:
    """Save the chunk text that could not be parsed correctly, for debugging."""
    Path(config.FAILED_CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
    # Sanitize filename: keep only the stem and strip path separators just in case.
    safe_name = Path(filename).stem.replace("/", "_").replace("\\", "_")
    out_path = Path(config.FAILED_CHUNKS_DIR) / f"{safe_name}_{chunk_idx}.txt"
    out_path.write_text(chunk_content, encoding="utf-8")
    print(f"  [INFO] Failed chunk saved to: {out_path}")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def process_file(path: Path, use_judge: bool) -> list[dict[str, Any]]:
    """Process a single file and return the list of valid blocks."""
    print(f"\n[FILE] {path}")
    text = read_file(path)
    if not text.strip():
        print("  [SKIP] Empty or unreadable file.")
        return []

    tokens = count_tokens(text)
    print(f"  Estimated tokens: {tokens}")

    if tokens <= config.MAX_TOKENS_DOC:
        chunks = [text]
        print("  Strategy: WHOLE document (under threshold).")
    else:
        chunks = chunk_text(text, config.CHUNK_SIZE, config.OVERLAP)
        print(f"  Strategy: CHUNKING -> {len(chunks)} chunk(s)")

    blocks_out: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        print(f"  -> Chunk {idx + 1}/{len(chunks)} (LLM call)...")
        parsed = extract_blocks_from_chunk(chunk)

        if not parsed or "blocks" not in parsed or not isinstance(parsed["blocks"], list):
            print("  [WARN] Invalid LLM output — saving the chunk and moving on.")
            save_failed_chunk(path.name, idx, chunk)
            continue

        for block in parsed["blocks"]:
            if not validate_block(block):
                print("  [WARN] Block discarded (missing/invalid keys).")
                continue

            if use_judge:
                ok = judge_block(block["source_quote"], block["answer"])
                if not ok:
                    print("  [JUDGE] Block discarded (verdict NO).")
                    continue

            blocks_out.append({
                "id": str(uuid.uuid4()),
                "answer": block["answer"].strip(),
                "questions": [q.strip() for q in block["questions"] if q.strip()],
                "category": block.get("category", "").strip(),
                "source_file": str(path),
                "source_quote": block["source_quote"].strip(),
            })

    print(f"  -> Valid blocks extracted: {len(blocks_out)}")
    return blocks_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data.json from source/ using Gemini.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-a-Judge to filter blocks not supported by the source.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FAQ Retrieval — Script 1: prepare_data.py")
    print(f"  LLM model     : {config.LLM_MODEL}")
    print(f"  Source dir    : {config.SOURCE_DIR}")
    print(f"  Output        : {config.DATA_JSON}")
    print(f"  Judge enabled : {args.judge}")
    print("=" * 70)

    files = discover_files(config.SOURCE_DIR)
    if not files:
        print(f"[ERROR] No supported files found in {config.SOURCE_DIR}/")
        sys.exit(1)

    print(f"\nFound {len(files)} file(s).")

    all_blocks: list[dict[str, Any]] = []
    for path in tqdm(files, desc="Files", unit="file"):
        blocks = process_file(path, use_judge=args.judge)
        all_blocks.extend(blocks)

    # Always overwrites data.json
    with open(config.DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(all_blocks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE. Total blocks: {len(all_blocks)}")
    print(f"Output saved to: {config.DATA_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
