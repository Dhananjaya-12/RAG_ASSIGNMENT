# Adaptive RAG Inference System

A self-optimizing Retrieval-Augmented Generation (RAG) pipeline that adapts its retrieval strategy, depth, and parameters at inference time based on query complexity and historical performance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Semantic Cache      │──── Cache Hit ──→ Return
                    │  (cosine sim ≥ 0.92)  │
                    └──────────┬───────────┘
                               │ Cache Miss
                               ▼
                    ┌──────────────────────┐     ┌─────────────────┐
                    │  Adaptive Decision   │◄────│  Feedback Loop  │
                    │      Layer           │     │  (K adjustment, │
                    │                      │     │   strategy hint) │
                    │  • Query complexity  │     └─────────────────┘
                    │  • Word count        │
                    │  • Indicator words   │
                    │  • Latency budget    │
                    └──────────┬───────────┘
                               │ Params: K, strategy, reranker, decomposition
                               ▼
                    ┌──────────────────────┐
                    │  Query Decomposer    │  (if complex → split into sub-queries)
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌─────────────────┐ ┌────────────────┐ ┌────────────────┐
    │  FAISS Vector   │ │  BM25 Keyword  │ │  (Sub-query    │
    │    Search       │ │    Search      │ │   retrieval)   │
    │  (semantic)     │ │  (exact match) │ │                │
    └────────┬────────┘ └───────┬────────┘ └────────────────┘
             │                  │
             └────────┬─────────┘
                      ▼
           ┌──────────────────────┐
           │  Reciprocal Rank     │
           │  Fusion (RRF)        │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  Cross-Encoder       │  (optional, skipped for simple queries)
           │  Re-Ranker           │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  LLM Generator       │
           │  (Gemini 2.0 Flash   │
           │   or Mock)           │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  Feedback Loop       │
           │  • Record latency    │
           │  • Compute quality   │
           │  • Adjust params     │
           └──────────┬───────────┘
                      ▼
              ┌───────────────┐
              │   RESPONSE    │
              │  + sources    │
              │  + metrics    │
              └───────────────┘
```

## Design Decisions

### 1. Embedding Model: `all-MiniLM-L6-v2`
- **Why**: Lightweight (80MB), fast inference, 384-dim vectors — good balance of quality and speed for a mini RAG system.
- **Tradeoff**: Larger models (e.g., `all-mpnet-base-v2`) offer better semantic understanding but are 3x slower.

### 2. FAISS with Inner Product (cosine similarity)
- **Why**: FAISS is the industry standard for vector search. Using `IndexFlatIP` with normalized vectors gives exact cosine similarity.
- **Tradeoff**: For millions of documents, approximate indexes (IVF, HNSW) would be needed. Flat index is fine for our scale.

### 3. Hybrid Retrieval with RRF
- **Why**: Vector search excels at semantic similarity but misses exact keyword matches. BM25 catches those. Reciprocal Rank Fusion combines both without needing to normalize scores.
- **Tradeoff**: Hybrid retrieval is slower than vector-only. The adaptive layer handles this by using vector-only for simple queries.

### 4. Cross-Encoder Re-Ranking
- **Why**: Bi-encoder (embedding) similarity is approximate. Cross-encoder scores query-document pairs directly, providing much more accurate relevance ranking.
- **Tradeoff**: Cross-encoders are ~10x slower than bi-encoders. Only used for medium/complex queries.

### 5. Rule-Based Adaptive Logic (No ML Training)
- **Why**: Simple, interpretable, and debuggable. Multi-factor scoring considers word count, complexity indicators, and historical performance.
- **Tradeoff**: A learned model could be more accurate, but requires training data we don't have.

### 6. Quality Proxy (Heuristic)
- **Why**: Without human judgments, we use a multi-factor heuristic: answer length, query-answer keyword overlap, retrieval score strength, and error detection.
- **Tradeoff**: Not as accurate as human evaluation or LLM-as-judge, but sufficient for adaptive adjustments.

## Tradeoffs

| Decision | Benefit | Cost |
|----------|---------|------|
| Adaptive K | Faster for simple queries, better for complex | Slight overhead for complexity analysis |
| Hybrid retrieval | Better recall than vector-only | ~2x retrieval time |
| Cross-encoder re-ranking | Higher precision | ~100-200ms per query |
| Semantic cache | Near-instant for repeated queries | Memory usage, potential staleness |
| Query decomposition | Better multi-part answers | More retrieval calls |

## Setup

### Prerequisites
- Python 3.9+
- (Optional) Google Gemini API key for LLM generation

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env and add your Google API key (or set LLM_PROVIDER=mock)
```

### Quick Start

```bash
# 1. Ingest documents and build index
python main.py ingest

# 2. Run a demo with sample queries
python main.py demo

# 3. Interactive query mode
python main.py interactive

# 4. Single query
python main.py query "What is machine learning?"

# 5. Run benchmarks
python main.py benchmark
```

## Project Structure

```
├── config.py                  # Central configuration
├── main.py                    # CLI entry point
├── src/
│   ├── ingestion.py           # Document loading & chunking
│   ├── embeddings.py          # Sentence-transformer embeddings
│   ├── vector_store.py        # FAISS index management
│   ├── keyword_search.py      # BM25 keyword search
│   ├── hybrid_retriever.py    # RRF fusion + re-ranking
│   ├── reranker.py            # Cross-encoder re-ranker
│   ├── generator.py           # LLM generation (Gemini/mock)
│   ├── adaptive_layer.py      # Query complexity analysis
│   ├── feedback_loop.py       # Performance tracking & adjustment
│   ├── cache.py               # Semantic caching
│   ├── query_decomposer.py    # Multi-step query decomposition
│   └── pipeline.py            # Full pipeline orchestrator
├── data/sample_documents/     # Sample documents for testing
├── benchmarks/                # Performance measurement
└── reports/                   # Analysis reports
```

## How the System Adapts

### At Query Time (Adaptive Layer)
1. **Query Analysis**: Each query is scored for complexity (0-1) based on word count, indicator words, and structure.
2. **Parameter Selection**: Complexity maps to retrieval parameters:
   - **LOW** (score < 0.3): K=3, vector-only, no re-ranker → fast
   - **MEDIUM** (0.3-0.6): K=5, hybrid, with re-ranker → balanced
   - **HIGH** (score > 0.6): K=10, hybrid, re-ranker + decomposition → thorough

### Over Time (Feedback Loop)
1. **Metric Tracking**: Every query records latency, retrieval time, generation time, and quality score.
2. **Adaptive Rules**: Based on a rolling window of recent queries:
   - Low quality → increase K (+2)
   - High quality → decrease K (-1) for efficiency
   - High latency → reduce K, switch to vector-only
   - Good performance → let adaptive layer decide freely

### Caching (Bonus)
- Semantically similar queries (cosine sim ≥ 0.92) return cached results instantly.
- LRU eviction with hit-count boosting for popular queries.

