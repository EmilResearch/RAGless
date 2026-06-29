"""
config.py
---------
Centralized configuration for RAGless.
All "magic" constants of the system live here to make tuning and
maintenance easier.
"""

# =============================================================================
# LLM & EMBEDDING MODELS (via LiteLLM)
# =============================================================================
# Note: model names follow LiteLLM's "provider/model" convention.
LLM_MODEL = "gemini/gemini-2.5-flash"
EMBEDDING_MODEL = "gemini/gemini-embedding-001"

# Output dimension of the embedding vector.
# gemini-embedding-001 supports Matryoshka truncation; 3072 is the native size.
VECTOR_SIZE = 3072

# Task type passed to the Gemini embedding API.
# Using asymmetric task types (DOCUMENT for ingestion, QUERY for retrieval)
# significantly improves retrieval quality with Gemini embeddings.
EMBED_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# =============================================================================
# QDRANT (local embedded mode — no server, no Docker)
# =============================================================================
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "qa_knowledge_base"

# =============================================================================
# CHUNKING & TOKEN BUDGET (Script 1 — prepare_data.py)
# =============================================================================
# If a document has <= MAX_TOKENS_DOC tokens, it is sent whole to the LLM.
# Otherwise it is split into CHUNK_SIZE-token chunks with OVERLAP tokens of overlap.
MAX_TOKENS_DOC = 10_000
CHUNK_SIZE = 8_000
OVERLAP = 500

# =============================================================================
# RETRIEVAL (Script 3 — chatbot.py)
# =============================================================================
# How many candidate matches we pull from Qdrant before aggregating by answer_id.
TOP_K_RETRIEVAL = 10

# Minimum *aggregated* score required to consider the answer reliable (OOD guard).
DEFAULT_THRESHOLD = 0.70

# Fallback threshold on the *best single* hit score. If the aggregated score is
# below DEFAULT_THRESHOLD but a single variant matches strongly, we still answer.
# This avoids false negatives when only one question variant of an answer matches.
SINGLE_HIT_THRESHOLD = 0.82

# =============================================================================
# EMBEDDING BATCHING (Script 2 — ingest_to_qdrant.py)
# =============================================================================
# Number of questions sent per embedding API call.
EMBEDDING_BATCH_SIZE = 100

# Number of points sent per Qdrant upsert call.
QDRANT_UPSERT_BATCH = 256

# =============================================================================
# RESILIENCE / RETRY
# =============================================================================
# Number of automatic retries (exponential backoff handled internally by LiteLLM)
# to cope with rate limits and transient errors from the Gemini API.
LITELLM_NUM_RETRIES = 3

# =============================================================================
# FILE PATHS
# =============================================================================
SOURCE_DIR = "./source"
DATA_JSON = "./data.json"
FAILED_CHUNKS_DIR = "./failed_chunks"
MISSED_QUERIES_LOG = "./missed_queries.log"
