"""
Semantic Cache Module
Caches query-result pairs with semantic similarity matching,
so similar queries can reuse previous results without re-computation.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """A cached query-result pair."""
    query: str
    query_embedding: List[float]  # stored as list for JSON serialization
    result: Dict
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    """
    LRU cache with semantic similarity matching.
    
    Instead of exact string matching, this cache uses cosine similarity
    between query embeddings to find cache hits. This means semantically
    similar questions (e.g., "What is ML?" and "Define machine learning")
    can share cached results.
    """
    
    def __init__(
        self,
        embedding_model,
        max_size: int = 100,
        similarity_threshold: float = 0.92,
        cache_path: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache_path = cache_path
        
        self.entries: List[CacheEntry] = []
        self._stats = {"hits": 0, "misses": 0}
        
        # Load persisted cache
        if cache_path:
            self._load_cache(cache_path)
    
    def get(self, query: str) -> Optional[Dict]:
        """
        Check if a semantically similar query exists in cache.
        
        Args:
            query: The search query
        
        Returns:
            Cached result dict if found, None otherwise
        """
        if not self.entries:
            self._stats["misses"] += 1
            return None
        
        query_embedding = self.embedding_model.encode(query).flatten()
        
        best_score = -1.0
        best_entry = None
        
        for entry in self.entries:
            cached_embedding = np.array(entry.query_embedding, dtype=np.float32)
            score = float(np.dot(query_embedding, cached_embedding))
            
            if score > best_score:
                best_score = score
                best_entry = entry
        
        if best_entry and best_score >= self.similarity_threshold:
            # Cache hit!
            best_entry.hit_count += 1
            self._stats["hits"] += 1
            
            result = best_entry.result.copy()
            result["cache_hit"] = True
            result["cache_similarity"] = best_score
            result["cached_query"] = best_entry.query
            
            return result
        
        self._stats["misses"] += 1
        return None
    
    def put(self, query: str, result: Dict):
        """
        Store a query-result pair in the cache.
        
        Args:
            query: The search query
            result: The full result dict to cache
        """
        query_embedding = self.embedding_model.encode(query).flatten()
        
        # Check if a very similar entry already exists (update it)
        for i, entry in enumerate(self.entries):
            cached_embedding = np.array(entry.query_embedding, dtype=np.float32)
            score = float(np.dot(query_embedding, cached_embedding))
            if score >= 0.98:  # Near-duplicate → update
                self.entries[i] = CacheEntry(
                    query=query,
                    query_embedding=query_embedding.tolist(),
                    result=self._make_serializable(result),
                    timestamp=time.time(),
                    hit_count=entry.hit_count,
                )
                return
        
        # Evict oldest entry if at capacity
        if len(self.entries) >= self.max_size:
            # LRU: remove the entry with the oldest timestamp and lowest hit count
            self.entries.sort(key=lambda e: (e.hit_count, e.timestamp))
            self.entries.pop(0)
        
        # Add new entry
        self.entries.append(CacheEntry(
            query=query,
            query_embedding=query_embedding.tolist(),
            result=self._make_serializable(result),
            timestamp=time.time(),
        ))
        
        # Persist
        if self.cache_path:
            self._save_cache(self.cache_path)
    
    def get_stats(self) -> Dict:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "entries": len(self.entries),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": self._stats["hits"] / max(total, 1),
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.entries.clear()
        self._stats = {"hits": 0, "misses": 0}
    
    @staticmethod
    def _make_serializable(result: Dict) -> Dict:
        """Ensure result dict is JSON-serializable."""
        clean = {}
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                clean[key] = value
            elif isinstance(value, (list, tuple)):
                clean[key] = [
                    v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                    for v in value
                ]
            elif isinstance(value, dict):
                clean[key] = SemanticCache._make_serializable(value)
            else:
                clean[key] = str(value)
        return clean
    
    def _save_cache(self, path: str):
        """Persist cache to disk."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "entries": [
                {
                    "query": entry.query,
                    "query_embedding": entry.query_embedding,
                    "result": entry.result,
                    "timestamp": entry.timestamp,
                    "hit_count": entry.hit_count,
                }
                for entry in self.entries
            ],
            "stats": self._stats,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def _load_cache(self, path: str):
        """Load cache from disk."""
        filepath = Path(path)
        if not filepath.exists():
            return
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data.get("entries", []):
                self.entries.append(CacheEntry(
                    query=item["query"],
                    query_embedding=item["query_embedding"],
                    result=item["result"],
                    timestamp=item["timestamp"],
                    hit_count=item.get("hit_count", 0),
                ))
            self._stats = data.get("stats", {"hits": 0, "misses": 0})
        except Exception as e:
            print(f"  [WARNING] Could not load cache: {e}")
