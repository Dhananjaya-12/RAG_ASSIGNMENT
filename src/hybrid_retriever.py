"""
Hybrid Retriever Module
Combines vector search and BM25 keyword search using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Optional
from .vector_store import VectorStore
from .keyword_search import KeywordSearch
from .reranker import Reranker, SimpleReranker
from .embeddings import EmbeddingModel


class HybridRetriever:
    """
    Hybrid retrieval engine that combines:
    1. FAISS vector search (semantic similarity)
    2. BM25 keyword search (exact term matching)
    3. Reciprocal Rank Fusion (RRF) for score combination
    4. Optional cross-encoder re-ranking
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_search: KeywordSearch,
        embedding_model: EmbeddingModel,
        reranker: Optional[Reranker] = None,
        simple_reranker: Optional[SimpleReranker] = None,
        hybrid_alpha: float = 0.6,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.simple_reranker = simple_reranker or SimpleReranker()
        self.hybrid_alpha = hybrid_alpha
        self.rrf_k = rrf_k
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        use_reranker: bool = True,
        reranker_top_n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents using the specified strategy.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            strategy: "vector_only", "keyword_only", or "hybrid"
            use_reranker: Whether to apply re-ranking
            reranker_top_n: Number of results after re-ranking (defaults to top_k)
        
        Returns:
            List of retrieved document dicts with scores
        """
        reranker_top_n = reranker_top_n or top_k
        
        if strategy == "vector_only":
            results = self._vector_search(query, top_k)
        elif strategy == "keyword_only":
            results = self._keyword_search(query, top_k)
        elif strategy == "hybrid":
            results = self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        
        # Apply re-ranking if enabled and we have results
        if use_reranker and results:
            results = self._rerank(query, results, reranker_top_n)
        
        return results
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform vector similarity search."""
        query_embedding = self.embedding_model.encode(query)
        results = self.vector_store.search(query_embedding, top_k)
        
        # Tag results with retrieval method
        for r in results:
            r["retrieval_method"] = "vector"
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform BM25 keyword search."""
        results = self.keyword_search.search(query, top_k)
        
        for r in results:
            r["retrieval_method"] = "keyword"
        
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Combine vector and keyword search using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = Σ 1 / (k + rank(d))
        where k is a constant (typically 60) and rank is 1-indexed.
        """
        # Get more candidates than needed for better fusion
        fetch_k = min(top_k * 3, max(top_k + 10, 15))
        
        vector_results = self._vector_search(query, fetch_k)
        keyword_results = self._keyword_search(query, fetch_k)
        
        # Build RRF scores
        rrf_scores: Dict[int, float] = {}
        doc_data: Dict[int, Dict] = {}
        
        # Score from vector results
        for rank, result in enumerate(vector_results, start=1):
            idx = result["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + self.hybrid_alpha / (self.rrf_k + rank)
            doc_data[idx] = result
        
        # Score from keyword results
        for rank, result in enumerate(keyword_results, start=1):
            idx = result["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - self.hybrid_alpha) / (self.rrf_k + rank)
            if idx not in doc_data:
                doc_data[idx] = result
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        
        results = []
        for idx in sorted_indices[:top_k]:
            result = doc_data[idx].copy()
            result["score"] = rrf_scores[idx]
            result["retrieval_method"] = "hybrid"
            results.append(result)
        
        return results
    
    def _rerank(self, query: str, documents: List[Dict], top_n: int) -> List[Dict]:
        """Apply re-ranking to the retrieved documents."""
        try:
            if self.reranker is not None:
                return self.reranker.rerank(query, documents, top_n)
        except Exception as e:
            print(f"  [WARNING] Cross-encoder re-ranking failed: {e}")
            print(f"  Falling back to simple re-ranker.")
        
        # Fallback to simple re-ranker
        return self.simple_reranker.rerank(query, documents, top_n)
