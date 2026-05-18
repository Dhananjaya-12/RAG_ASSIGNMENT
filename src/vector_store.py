"""
FAISS Vector Store Module
Manages the FAISS index for efficient approximate nearest neighbor search.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")


class VectorStore:
    """
    FAISS-based vector store for document embeddings.
    Uses Inner Product index with L2-normalized vectors for cosine similarity.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim when normalized)
        self.metadata: List[Dict] = []  # Parallel metadata for each vector
    
    def add(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Add embeddings and their metadata to the index.
        
        Args:
            embeddings: numpy array of shape (n, dimension), should be L2-normalized
            metadata_list: list of metadata dicts, one per embedding
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Ensure normalized for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for the most similar vectors.
        
        Args:
            query_embedding: numpy array of shape (1, dimension) or (dimension,)
            top_k: number of results to return
        
        Returns:
            List of dicts with 'score', 'index', and all metadata fields
        """
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Clamp top_k to available vectors
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
        
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = {
                "score": float(score),
                "index": int(idx),
            }
            if idx < len(self.metadata):
                result.update(self.metadata[idx])
            results.append(result)
        
        return results
    
    @property
    def total_vectors(self) -> int:
        """Return the total number of vectors in the index."""
        return self.index.ntotal
    
    def save(self, index_path: str, metadata_path: str):
        """Save the FAISS index and metadata to disk."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        
        # Create directories if needed
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "metadata": self.metadata,
            }, f, indent=2)
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "VectorStore":
        """Load a FAISS index and metadata from disk."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index files not found at {index_path}")
        
        index = faiss.read_index(str(index_path))
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        store = cls(dimension=data["dimension"])
        store.index = index
        store.metadata = data["metadata"]
        
        return store
    
    def exists(self, index_path: str, metadata_path: str) -> bool:
        """Check if saved index files exist."""
        return Path(index_path).exists() and Path(metadata_path).exists()
