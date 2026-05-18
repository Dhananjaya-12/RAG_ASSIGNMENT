"""
Pipeline Orchestrator
Ties all components together into a complete Adaptive RAG pipeline.
"""

import time
from typing import Dict, Optional, List
from pathlib import Path

from .ingestion import load_documents, chunk_documents, get_chunk_texts, get_chunk_metadata
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .keyword_search import KeywordSearch
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker, SimpleReranker
from .generator import get_generator, format_prompt
from .adaptive_layer import AdaptiveLayer
from .feedback_loop import FeedbackLoop, QueryMetrics
from .cache import SemanticCache
from .query_decomposer import QueryDecomposer


class AdaptiveRAGPipeline:
    """
    Complete Adaptive RAG Pipeline that orchestrates:
    1. Document ingestion and indexing
    2. Hybrid retrieval (vector + BM25 + re-ranking)
    3. Adaptive parameter selection based on query complexity
    4. LLM-based answer generation
    5. Feedback-driven optimization
    6. Semantic caching
    7. Query decomposition for complex queries
    """
    
    def __init__(self, config):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration module with all parameters
        """
        self.config = config
        self.is_initialized = False
        
        # Core components (initialized lazily)
        self.embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
        self.vector_store = None
        self.keyword_search = KeywordSearch()
        self.reranker = Reranker(config.RERANKER_MODEL)
        self.simple_reranker = SimpleReranker()
        self.retriever = None
        self.generator = get_generator(config.LLM_PROVIDER)
        
        # Adaptive components
        self.adaptive_layer = AdaptiveLayer(
            default_top_k=config.DEFAULT_TOP_K,
            min_top_k=config.MIN_TOP_K,
            max_top_k=config.MAX_TOP_K,
            short_query_words=config.SHORT_QUERY_WORDS,
            latency_threshold=config.LATENCY_SLA_SECONDS,
        )
        
        self.feedback_loop = FeedbackLoop(
            window_size=config.FEEDBACK_WINDOW_SIZE,
            latency_sla=config.LATENCY_SLA_SECONDS,
            quality_low_threshold=config.QUALITY_LOW_THRESHOLD,
            quality_high_threshold=config.QUALITY_HIGH_THRESHOLD,
            history_path=str(config.FEEDBACK_HISTORY_PATH),
        )
        
        # Bonus: Cache
        self.cache = None
        if config.CACHE_ENABLED:
            self.cache = SemanticCache(
                embedding_model=self.embedding_model,
                max_size=config.CACHE_MAX_SIZE,
                similarity_threshold=config.CACHE_SIMILARITY_THRESHOLD,
                cache_path=str(config.CACHE_PATH),
            )
        
        # Bonus: Query decomposer
        self.query_decomposer = QueryDecomposer()
        
        # Document chunks (for reference)
        self.chunks = []
    
    def ingest(self, data_dir: Optional[str] = None) -> Dict:
        """
        Ingest documents, create embeddings, and build indexes.
        
        Args:
            data_dir: Path to document directory (uses config default if None)
        
        Returns:
            Dict with ingestion statistics
        """
        data_dir = data_dir or str(self.config.DATA_DIR)
        
        print("\n" + "=" * 60)
        print("  DOCUMENT INGESTION")
        print("=" * 60)
        
        # Step 1: Load documents
        print(f"\n  Loading documents from: {data_dir}")
        documents = load_documents(data_dir, self.config.SUPPORTED_EXTENSIONS)
        print(f"  Loaded {len(documents)} documents")
        
        if not documents:
            raise ValueError(f"No documents found in {data_dir}")
        
        # Step 2: Chunk documents
        print(f"  Chunking (size={self.config.CHUNK_SIZE}, overlap={self.config.CHUNK_OVERLAP})...")
        self.chunks = chunk_documents(
            documents,
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )
        print(f"  Created {len(self.chunks)} chunks")
        
        # Step 3: Generate embeddings
        print(f"  Generating embeddings...")
        t0 = time.time()
        chunk_texts = get_chunk_texts(self.chunks)
        embeddings = self.embedding_model.encode(chunk_texts, show_progress=True)
        embed_time = time.time() - t0
        print(f"  Embeddings generated in {embed_time:.2f}s")
        
        # Step 4: Build FAISS index
        print(f"  Building FAISS index...")
        metadata_list = get_chunk_metadata(self.chunks)
        self.vector_store = VectorStore(dimension=self.embedding_model.dimension)
        self.vector_store.add(embeddings, metadata_list)
        print(f"  FAISS index built: {self.vector_store.total_vectors} vectors")
        
        # Step 5: Build BM25 index
        print(f"  Building BM25 keyword index...")
        self.keyword_search.build_index(metadata_list)
        print(f"  BM25 index built")
        
        # Step 6: Create hybrid retriever
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            keyword_search=self.keyword_search,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            simple_reranker=self.simple_reranker,
            hybrid_alpha=self.config.HYBRID_ALPHA,
            rrf_k=self.config.RRF_K,
        )
        
        # Step 7: Save index
        print(f"  Saving index to disk...")
        self.vector_store.save(
            str(self.config.FAISS_INDEX_PATH),
            str(self.config.FAISS_METADATA_PATH),
        )
        print(f"  Index saved")
        
        self.is_initialized = True
        
        stats = {
            "documents_loaded": len(documents),
            "chunks_created": len(self.chunks),
            "embedding_time": embed_time,
            "index_size": self.vector_store.total_vectors,
            "sources": [doc["source"] for doc in documents],
        }
        
        print(f"\n  Ingestion complete!")
        print(f"  Documents: {stats['documents_loaded']}")
        print(f"  Chunks: {stats['chunks_created']}")
        print(f"  Embedding time: {stats['embedding_time']:.2f}s")
        print("=" * 60)
        
        return stats
    
    def load_index(self) -> bool:
        """
        Load a previously built index from disk.
        
        Returns:
            True if index was loaded successfully
        """
        try:
            index_path = str(self.config.FAISS_INDEX_PATH)
            metadata_path = str(self.config.FAISS_METADATA_PATH)
            
            if not Path(index_path).exists():
                return False
            
            print("  Loading saved index...")
            self.vector_store = VectorStore.load(index_path, metadata_path)
            
            # Rebuild BM25 from metadata
            self.keyword_search.build_index(self.vector_store.metadata)
            
            # Create retriever
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                keyword_search=self.keyword_search,
                embedding_model=self.embedding_model,
                reranker=self.reranker,
                simple_reranker=self.simple_reranker,
                hybrid_alpha=self.config.HYBRID_ALPHA,
                rrf_k=self.config.RRF_K,
            )
            
            self.is_initialized = True
            print(f"  Index loaded: {self.vector_store.total_vectors} vectors")
            return True
        except Exception as e:
            print(f"  [WARNING] Could not load index: {e}")
            return False
    
    def query(
        self,
        question: str,
        use_adaptive: bool = True,
        use_cache: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Process a query through the full adaptive RAG pipeline.
        
        Flow:
        1. Check cache for similar previous query
        2. Analyze query complexity (adaptive layer)
        3. Apply feedback loop adjustments
        4. Decompose query if needed
        5. Retrieve relevant documents
        6. Generate answer with LLM
        7. Track metrics and update feedback loop
        8. Cache result
        
        Args:
            question: The user's question
            use_adaptive: Whether to use adaptive parameter selection
            use_cache: Whether to check/update cache
            verbose: Whether to print detailed info
        
        Returns:
            Dict with answer, sources, metrics, and adaptive decisions
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call ingest() or load_index() first.")
        
        total_start = time.time()
        
        # ─── Step 1: Cache Check ─────────────────────────────────────
        if use_cache and self.cache:
            cached = self.cache.get(question)
            if cached:
                if verbose:
                    print(f"  CACHE HIT (similarity: {cached.get('cache_similarity', 0):.3f})")
                cached["total_latency"] = time.time() - total_start
                return cached
        
        # ─── Step 2: Adaptive Parameter Selection ────────────────────
        feedback_adj = self.feedback_loop.get_adjustment()
        
        if use_adaptive:
            params = self.adaptive_layer.analyze_query(
                query=question,
                avg_latency=self._get_avg_latency(),
                feedback_adjustment=feedback_adj["k_adjustment"],
            )
        else:
            # Fixed parameters (non-adaptive mode for benchmarking)
            from .adaptive_layer import RetrievalParams
            params = RetrievalParams(
                top_k=self.config.DEFAULT_TOP_K,
                strategy="hybrid",
                use_reranker=True,
                use_decomposition=False,
                complexity="FIXED",
                complexity_score=0.5,
                reasoning="Non-adaptive mode (fixed parameters)",
            )
        
        # Apply strategy override from feedback
        if feedback_adj.get("strategy_override") and use_adaptive:
            params.strategy = feedback_adj["strategy_override"]
        
        if verbose:
            print(f"  Adaptive: {params.reasoning}")
        
        # ─── Step 3: Query Decomposition ─────────────────────────────
        sub_queries = [question]
        if params.use_decomposition:
            sub_queries = self.query_decomposer.decompose(question)
            if verbose and len(sub_queries) > 1:
                print(f"  Decomposed into {len(sub_queries)} sub-queries:")
                for i, sq in enumerate(sub_queries):
                    print(f"    {i+1}. {sq}")
        
        # ─── Step 4: Retrieval ───────────────────────────────────────
        retrieval_start = time.time()
        
        if len(sub_queries) == 1:
            retrieved = self.retriever.retrieve(
                query=question,
                top_k=params.top_k,
                strategy=params.strategy,
                use_reranker=params.use_reranker,
                reranker_top_n=self.config.RERANKER_TOP_N,
            )
        else:
            # Multi-query retrieval
            all_results = []
            for sq in sub_queries:
                results = self.retriever.retrieve(
                    query=sq,
                    top_k=params.top_k,
                    strategy=params.strategy,
                    use_reranker=params.use_reranker,
                    reranker_top_n=self.config.RERANKER_TOP_N,
                )
                all_results.append(results)
            retrieved = self.query_decomposer.merge_results(all_results)[:params.top_k]
        
        retrieval_time = time.time() - retrieval_start
        
        if verbose:
            print(f"  Retrieved {len(retrieved)} chunks in {retrieval_time:.3f}s")
            print(f"  Strategy: {params.strategy} | K={params.top_k} | Reranker={'ON' if params.use_reranker else 'OFF'}")
        
        # ─── Step 5: Build Context & Generate ────────────────────────
        context = self._build_context(retrieved)
        prompt = format_prompt(self.config.RAG_PROMPT_TEMPLATE, context, question)
        
        generation_start = time.time()
        response = self.generator.generate(prompt)
        generation_time = time.time() - generation_start
        
        total_latency = time.time() - total_start
        
        if verbose:
            print(f"  Generated answer in {generation_time:.3f}s (total: {total_latency:.3f}s)")
        
        # ─── Step 6: Compute Quality & Record Metrics ────────────────
        retrieval_scores = [r.get("score", 0) for r in retrieved]
        quality_score = self.feedback_loop.compute_quality_score(
            query=question,
            answer=response.get("text", ""),
            retrieval_scores=retrieval_scores,
        )
        
        metrics = QueryMetrics(
            timestamp=time.time(),
            query=question,
            total_latency=total_latency,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            answer_length=len(response.get("text", "")),
            quality_score=quality_score,
            top_k_used=params.top_k,
            strategy_used=params.strategy,
            use_reranker=params.use_reranker,
            complexity=params.complexity,
            num_results=len(retrieved),
        )
        self.feedback_loop.record(metrics)
        
        # ─── Step 7: Build Result ────────────────────────────────────
        result = {
            "answer": response.get("text", ""),
            "sources": [
                {
                    "file": r.get("source_file", "unknown"),
                    "chunk_index": r.get("chunk_index", 0),
                    "score": r.get("score", 0),
                    "preview": r.get("text", "")[:150] + "...",
                }
                for r in retrieved
            ],
            "metrics": {
                "total_latency": total_latency,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "quality_score": quality_score,
            },
            "adaptive": {
                "complexity": params.complexity,
                "complexity_score": params.complexity_score,
                "top_k": params.top_k,
                "strategy": params.strategy,
                "use_reranker": params.use_reranker,
                "reasoning": params.reasoning,
                "feedback_adjustment": feedback_adj,
            },
            "model": response.get("model", "unknown"),
            "cache_hit": False,
        }
        
        # ─── Step 8: Cache Result ────────────────────────────────────
        if use_cache and self.cache:
            self.cache.put(question, result)
        
        return result
    
    def _build_context(self, retrieved: List[Dict]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            source = doc.get("source_file", "unknown")
            text = doc.get("text", "")
            score = doc.get("score", 0)
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.3f})]\n{text}"
            )
        return "\n\n---\n\n".join(context_parts)
    
    def _get_avg_latency(self) -> float:
        """Get average latency from recent queries."""
        if not self.feedback_loop.all_latencies:
            return 0.0
        recent = self.feedback_loop.all_latencies[-10:]
        return sum(recent) / len(recent)
    
    def get_stats(self) -> Dict:
        """Get comprehensive pipeline statistics."""
        stats = {
            "pipeline": {
                "initialized": self.is_initialized,
                "index_size": self.vector_store.total_vectors if self.vector_store else 0,
                "llm_provider": self.config.LLM_PROVIDER,
            },
            "feedback": self.feedback_loop.get_statistics(),
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
