"""
Embedding Generation Module
Wraps sentence-transformers for efficient text embedding with lazy model loading.
"""

import numpy as np
from typing import List, Union, Optional


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for generating text embeddings.
    Uses lazy loading to avoid loading the model until first use.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            print(f"  Model loaded. Embedding dimension: {self._dimension}")
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            _ = self.model  # triggers loading
        return self._dimension
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            show_progress: Show progress bar during encoding
        
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        
        return np.array(embeddings, dtype=np.float32)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        Assumes embeddings are already L2-normalized.
        """
        return float(np.dot(embedding1.flatten(), embedding2.flatten()))
