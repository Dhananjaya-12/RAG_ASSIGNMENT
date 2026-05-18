"""
Adaptive Decision Layer Module
Analyzes query complexity at runtime and selects optimal retrieval parameters.
This is the "brain" of the adaptive RAG system.
"""

import re
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalParams:
    """Parameters selected by the adaptive layer for a given query."""
    top_k: int
    strategy: str          # "vector_only", "keyword_only", "hybrid"
    use_reranker: bool
    use_decomposition: bool
    complexity: str        # "LOW", "MEDIUM", "HIGH"
    complexity_score: float
    reasoning: str         # Human-readable explanation of the decision


class AdaptiveLayer:
    """
    Runtime decision layer that analyzes query characteristics
    and selects optimal retrieval parameters.
    
    Decision factors:
    1. Query length (word count)
    2. Query complexity (indicator words, structure)
    3. Historical performance (from feedback loop)
    4. Latency constraints
    """
    
    # Words that indicate complex queries requiring deeper retrieval
    COMPLEXITY_INDICATORS = {
        "high": [
            "compare", "contrast", "difference", "differences", "relationship",
            "analyze", "evaluate", "explain why", "how does", "elaborate",
            "discuss", "advantages", "disadvantages", "tradeoffs", "trade-offs",
            "implications", "impact", "versus", "vs", "between",
        ],
        "medium": [
            "explain", "describe", "how", "why", "what are", "list",
            "overview", "summary", "examples", "types", "categories",
        ],
    }
    
    def __init__(
        self,
        default_top_k: int = 5,
        min_top_k: int = 2,
        max_top_k: int = 15,
        short_query_words: int = 4,
        latency_threshold: float = 8.0,
    ):
        self.default_top_k = default_top_k
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self.short_query_words = short_query_words
        self.latency_threshold = latency_threshold
    
    def analyze_query(
        self,
        query: str,
        avg_latency: float = 0.0,
        feedback_adjustment: int = 0,
    ) -> RetrievalParams:
        """
        Analyze a query and determine optimal retrieval parameters.
        
        Args:
            query: The user's query string
            avg_latency: Average latency from recent queries (from feedback loop)
            feedback_adjustment: K adjustment from feedback loop (-2 to +2)
        
        Returns:
            RetrievalParams with selected settings and reasoning
        """
        # Step 1: Compute complexity score
        complexity_score = self._compute_complexity_score(query)
        
        # Step 2: Classify complexity
        if complexity_score < 0.3:
            complexity = "LOW"
        elif complexity_score < 0.6:
            complexity = "MEDIUM"
        else:
            complexity = "HIGH"
        
        # Step 3: Select base parameters from complexity
        params = self._get_base_params(complexity)
        
        # Step 4: Apply feedback loop adjustments
        params["top_k"] = max(
            self.min_top_k,
            min(self.max_top_k, params["top_k"] + feedback_adjustment)
        )
        
        # Step 5: Apply latency constraints
        reasoning_parts = [
            f"Query complexity: {complexity} (score: {complexity_score:.2f})",
        ]
        
        if avg_latency > self.latency_threshold:
            # High latency → reduce retrieval depth
            params["top_k"] = max(self.min_top_k, params["top_k"] - 2)
            params["use_reranker"] = False
            params["strategy"] = "vector_only"
            reasoning_parts.append(
                f"Latency constraint: avg {avg_latency:.1f}s > {self.latency_threshold}s threshold → reduced K and disabled re-ranker"
            )
        
        if feedback_adjustment != 0:
            reasoning_parts.append(
                f"Feedback adjustment: K adjusted by {feedback_adjustment:+d}"
            )
        
        reasoning_parts.append(
            f"Selected: K={params['top_k']}, strategy={params['strategy']}, "
            f"reranker={'ON' if params['use_reranker'] else 'OFF'}, "
            f"decomposition={'ON' if params['use_decomposition'] else 'OFF'}"
        )
        
        return RetrievalParams(
            top_k=params["top_k"],
            strategy=params["strategy"],
            use_reranker=params["use_reranker"],
            use_decomposition=params["use_decomposition"],
            complexity=complexity,
            complexity_score=complexity_score,
            reasoning=" | ".join(reasoning_parts),
        )
    
    def _compute_complexity_score(self, query: str) -> float:
        """
        Compute a complexity score between 0 and 1 for the query.
        
        Factors:
        - Word count
        - Presence of complexity indicator words
        - Number of question marks
        - Presence of conjunctions (multi-part queries)
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        word_count = len(words)
        
        score = 0.0
        
        # Factor 1: Word count (longer queries tend to be more complex)
        if word_count <= self.short_query_words:
            score += 0.1
        elif word_count <= 8:
            score += 0.25
        elif word_count <= 15:
            score += 0.4
        else:
            score += 0.5
        
        # Factor 2: Complexity indicator words
        high_indicators = sum(
            1 for phrase in self.COMPLEXITY_INDICATORS["high"]
            if phrase in query_lower
        )
        medium_indicators = sum(
            1 for phrase in self.COMPLEXITY_INDICATORS["medium"]
            if phrase in query_lower
        )
        
        score += min(0.3, high_indicators * 0.15)
        score += min(0.15, medium_indicators * 0.05)
        
        # Factor 3: Multiple question marks (multi-part question)
        question_marks = query.count("?")
        if question_marks > 1:
            score += 0.1
        
        # Factor 4: Conjunctions indicating multi-part queries
        conjunctions = sum(1 for w in words if w in {"and", "also", "additionally", "moreover", "furthermore"})
        score += min(0.1, conjunctions * 0.05)
        
        # Factor 5: Unique term diversity
        unique_ratio = len(set(words)) / max(word_count, 1)
        if unique_ratio > 0.8 and word_count > 5:
            score += 0.05  # High term diversity suggests complexity
        
        return min(1.0, score)
    
    def _get_base_params(self, complexity: str) -> Dict:
        """Get base retrieval parameters for a complexity level."""
        params_map = {
            "LOW": {
                "top_k": 3,
                "strategy": "vector_only",
                "use_reranker": False,
                "use_decomposition": False,
            },
            "MEDIUM": {
                "top_k": 5,
                "strategy": "hybrid",
                "use_reranker": True,
                "use_decomposition": False,
            },
            "HIGH": {
                "top_k": 10,
                "strategy": "hybrid",
                "use_reranker": True,
                "use_decomposition": True,
            },
        }
        return params_map.get(complexity, params_map["MEDIUM"]).copy()
