"""
BM25 Keyword Search Module
Implements keyword-based retrieval using the BM25 algorithm.
"""

import re
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


class KeywordSearch:
    """
    BM25-based keyword search engine.
    Complements vector search by capturing exact keyword matches
    that semantic embeddings might miss.
    """
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def build_index(self, metadata_list: List[Dict]):
        """
        Build BM25 index from document metadata.
        
        Args:
            metadata_list: List of dicts with 'text' key (same format as vector store metadata)
        """
        self.documents = metadata_list
        self.tokenized_corpus = [
            self._tokenize(doc.get("text", ""))
            for doc in metadata_list
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents matching the query using BM25.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
        
        Returns:
            List of dicts with 'score', 'index', and document metadata
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-K indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive-scoring results
                result = {
                    "score": float(scores[idx]),
                    "index": idx,
                }
                if idx < len(self.documents):
                    result.update(self.documents[idx])
                results.append(result)
        
        return results
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenization: lowercase, split on non-alphanumeric, remove stopwords.
        """
        # Basic stopwords (keep it lightweight, no NLTK dependency at runtime)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
            'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
            'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom',
        }
        
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        return tokens
