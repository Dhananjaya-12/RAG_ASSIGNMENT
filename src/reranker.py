"""
Cross-Encoder Re-Ranking Module
Uses a cross-encoder model to re-rank retrieved documents for higher relevance.
"""

from typing import List, Dict, Optional


class Reranker:
    """
    Cross-encoder based re-ranker that scores query-document pairs
    for more accurate relevance ranking than bi-encoder similarity alone.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"  Loading re-ranker model: {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
            print(f"  Re-ranker loaded.")
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Re-rank documents using cross-encoder scores.
        
        Args:
            query: The search query
            documents: List of document dicts (must have 'text' key)
            top_n: Number of top results to keep (None = keep all)
        
        Returns:
            Re-ranked list of document dicts with updated 'rerank_score'
        """
        if not documents:
            return []
        
        # Build query-document pairs
        pairs = [(query, doc.get("text", "")) for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Attach scores to documents
        reranked = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            reranked.append(doc_copy)
        
        # Sort by rerank score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Trim to top_n
        if top_n is not None:
            reranked = reranked[:top_n]
        
        return reranked


class SimpleReranker:
    """
    A lightweight re-ranker that doesn't require a cross-encoder model.
    Uses keyword overlap and position-based heuristics for re-ranking.
    Useful as a fallback when the cross-encoder is too slow.
    """
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """Re-rank using keyword overlap heuristics."""
        if not documents:
            return []
        
        query_terms = set(query.lower().split())
        
        reranked = []
        for doc in documents:
            doc_copy = doc.copy()
            text = doc.get("text", "").lower()
            doc_terms = set(text.split())
            
            # Compute keyword overlap
            overlap = len(query_terms & doc_terms)
            coverage = overlap / max(len(query_terms), 1)
            
            # Combine with original score
            original_score = doc.get("score", 0)
            doc_copy["rerank_score"] = 0.6 * original_score + 0.4 * coverage
            reranked.append(doc_copy)
        
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        if top_n is not None:
            reranked = reranked[:top_n]
        
        return reranked
