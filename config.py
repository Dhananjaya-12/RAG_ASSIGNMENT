"""
Central configuration for the Adaptive RAG Inference System.
All tunable parameters are defined here for easy experimentation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sample_documents"
INDEX_DIR = BASE_DIR / "indexes"
FEEDBACK_DIR = BASE_DIR / "feedback"
CACHE_DIR = BASE_DIR / "cache"

# ─── Document Ingestion ─────────────────────────────────────────────────────
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between chunks
SUPPORTED_EXTENSIONS = [".txt", ".md"]

# ─── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── FAISS Index ─────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
FAISS_METADATA_PATH = INDEX_DIR / "metadata.json"

# ─── Retrieval Defaults ─────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
MIN_TOP_K = 2
MAX_TOP_K = 15
HYBRID_ALPHA = 0.6        # weight for vector search (1-alpha for BM25)
RRF_K = 60                # Reciprocal Rank Fusion constant

# ─── Re-ranker ───────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N = 5        # number of results after re-ranking

# ─── LLM Generation ─────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "mock"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
MAX_GENERATION_TOKENS = 1024
TEMPERATURE = 0.3

# ─── Adaptive Layer ──────────────────────────────────────────────────────────
# Query complexity thresholds
SHORT_QUERY_WORDS = 4         # queries with <= this many words are "short"
COMPLEX_QUERY_INDICATORS = [
    "why", "how", "compare", "contrast", "explain", "difference",
    "relationship", "analyze", "evaluate", "discuss", "elaborate"
]
COMPLEXITY_THRESHOLDS = {
    "LOW": {"top_k": 3, "strategy": "vector_only", "use_reranker": False, "use_decomposition": False},
    "MEDIUM": {"top_k": 5, "strategy": "hybrid", "use_reranker": True, "use_decomposition": False},
    "HIGH": {"top_k": 10, "strategy": "hybrid", "use_reranker": True, "use_decomposition": True},
}

# ─── Feedback Loop ───────────────────────────────────────────────────────────
FEEDBACK_WINDOW_SIZE = 20     # rolling window of last N queries
LATENCY_SLA_SECONDS = 8.0    # target max latency
QUALITY_LOW_THRESHOLD = 0.35  # below this → increase K
QUALITY_HIGH_THRESHOLD = 0.7  # above this → allow K reduction
FEEDBACK_HISTORY_PATH = FEEDBACK_DIR / "history.json"

# ─── Cache ───────────────────────────────────────────────────────────────────
CACHE_ENABLED = True
CACHE_MAX_SIZE = 100
CACHE_SIMILARITY_THRESHOLD = 0.92  # cosine similarity for cache hit
CACHE_PATH = CACHE_DIR / "semantic_cache.json"

# ─── Prompt Template ─────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain enough information to fully answer the question, say so clearly.

Context:
{context}

Question: {question}

Provide a comprehensive, well-structured answer:"""
