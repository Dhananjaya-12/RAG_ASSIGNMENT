"""
Feedback Loop Module
Tracks query performance metrics and provides adaptive adjustments
to improve retrieval quality and latency over time.
No training required — pure rule-based adaptive logic.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class QueryMetrics:
    """Metrics tracked for a single query."""
    timestamp: float
    query: str
    total_latency: float        # seconds
    retrieval_time: float       # seconds
    generation_time: float      # seconds
    answer_length: int          # character count
    quality_score: float        # 0-1 heuristic quality proxy
    top_k_used: int
    strategy_used: str
    use_reranker: bool
    complexity: str
    num_results: int


class FeedbackLoop:
    """
    Tracks performance metrics and provides adaptive adjustments to the
    retrieval pipeline based on recent history.
    
    Adjustment rules (no ML training):
    - Quality too low → increase K
    - Quality consistently high → allow K reduction (efficiency)
    - Latency too high → reduce K, simplify strategy
    - Latency acceptable → allow more complex strategies
    """
    
    def __init__(
        self,
        window_size: int = 20,
        latency_sla: float = 8.0,
        quality_low_threshold: float = 0.35,
        quality_high_threshold: float = 0.7,
        history_path: Optional[str] = None,
    ):
        self.window_size = window_size
        self.latency_sla = latency_sla
        self.quality_low_threshold = quality_low_threshold
        self.quality_high_threshold = quality_high_threshold
        self.history_path = history_path
        
        # Rolling window of recent metrics
        self.history: deque = deque(maxlen=window_size)
        
        # Cumulative statistics
        self.total_queries = 0
        self.all_latencies: List[float] = []
        self.all_retrieval_times: List[float] = []
        self.all_generation_times: List[float] = []
        
        # Load persisted history if available
        if history_path:
            self._load_history(history_path)
    
    def record(self, metrics: QueryMetrics):
        """Record metrics for a completed query."""
        self.history.append(metrics)
        self.total_queries += 1
        self.all_latencies.append(metrics.total_latency)
        self.all_retrieval_times.append(metrics.retrieval_time)
        self.all_generation_times.append(metrics.generation_time)
        
        # Persist history
        if self.history_path:
            self._save_history(self.history_path)
    
    def compute_quality_score(
        self,
        query: str,
        answer: str,
        retrieval_scores: List[float],
    ) -> float:
        """
        Compute a heuristic quality proxy score between 0 and 1.
        
        Factors:
        1. Answer length (very short = likely poor quality)
        2. Query-answer keyword overlap (answer addresses the query)
        3. Retrieval score strength (high scores = better context)
        4. Answer has substance (not just an error or "I don't know")
        """
        if not answer or not answer.strip():
            return 0.0
        
        score = 0.0
        
        # Factor 1: Answer length (normalized 0-0.3)
        answer_len = len(answer)
        if answer_len < 50:
            score += 0.05
        elif answer_len < 150:
            score += 0.15
        elif answer_len < 500:
            score += 0.25
        else:
            score += 0.3
        
        # Factor 2: Query-answer keyword overlap (0-0.35)
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'in', 'for', 'on', 'with', 'it', 'this', 'that'}
        query_terms -= stopwords
        overlap = len(query_terms & answer_terms)
        coverage = overlap / max(len(query_terms), 1)
        score += min(0.35, coverage * 0.35)
        
        # Factor 3: Retrieval score strength (0-0.2)
        if retrieval_scores:
            avg_score = sum(retrieval_scores) / len(retrieval_scores)
            score += min(0.2, avg_score * 0.25)
        
        # Factor 4: Answer substance check (0-0.15)
        error_indicators = ["error", "i don't know", "no information", "cannot", "unable"]
        has_error = any(indicator in answer.lower() for indicator in error_indicators)
        if not has_error:
            score += 0.15
        
        return min(1.0, score)
    
    def get_adjustment(self) -> Dict:
        """
        Compute adaptive adjustments based on recent history.
        
        Returns:
            Dict with:
            - 'k_adjustment': int (-2 to +2) to add to the adaptive layer's K
            - 'strategy_override': Optional[str] strategy to force
            - 'reasoning': str explanation of the adjustment
        """
        if len(self.history) < 3:
            return {
                "k_adjustment": 0,
                "strategy_override": None,
                "reasoning": "Not enough history for adjustment (need >= 3 queries)"
            }
        
        recent = list(self.history)
        
        # Compute averages
        avg_latency = sum(m.total_latency for m in recent) / len(recent)
        avg_quality = sum(m.quality_score for m in recent) / len(recent)
        avg_k = sum(m.top_k_used for m in recent) / len(recent)
        
        k_adjustment = 0
        strategy_override = None
        reasoning_parts = []
        
        # Rule 1: Quality too low → increase K
        if avg_quality < self.quality_low_threshold:
            k_adjustment += 2
            reasoning_parts.append(
                f"Low quality ({avg_quality:.2f} < {self.quality_low_threshold}) → +2 K"
            )
        
        # Rule 2: Quality consistently high → reduce K for efficiency
        elif avg_quality > self.quality_high_threshold and avg_k > 4:
            k_adjustment -= 1
            reasoning_parts.append(
                f"High quality ({avg_quality:.2f} > {self.quality_high_threshold}) → -1 K for efficiency"
            )
        
        # Rule 3: High latency → reduce complexity
        if avg_latency > self.latency_sla:
            k_adjustment -= 2
            strategy_override = "vector_only"
            reasoning_parts.append(
                f"High latency ({avg_latency:.1f}s > {self.latency_sla}s SLA) → -2 K, vector-only"
            )
        
        # Rule 4: Low latency + good quality → allow hybrid
        elif avg_latency < self.latency_sla * 0.5 and avg_quality > 0.5:
            strategy_override = None  # Let adaptive layer decide
            reasoning_parts.append(
                f"Good latency ({avg_latency:.1f}s) and quality ({avg_quality:.2f}) → allow adaptive decisions"
            )
        
        if not reasoning_parts:
            reasoning_parts.append("No adjustment needed — performance within targets")
        
        return {
            "k_adjustment": k_adjustment,
            "strategy_override": strategy_override,
            "reasoning": " | ".join(reasoning_parts),
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive performance statistics."""
        if not self.all_latencies:
            return {"total_queries": 0}
        
        sorted_latencies = sorted(self.all_latencies)
        sorted_retrieval = sorted(self.all_retrieval_times)
        sorted_generation = sorted(self.all_generation_times)
        
        def percentile(data, p):
            if not data:
                return 0.0
            idx = int(len(data) * p / 100)
            idx = min(idx, len(data) - 1)
            return data[idx]
        
        recent = list(self.history)
        recent_quality = [m.quality_score for m in recent] if recent else [0]
        
        stats = {
            "total_queries": self.total_queries,
            "latency": {
                "mean": sum(self.all_latencies) / len(self.all_latencies),
                "p50": percentile(sorted_latencies, 50),
                "p95": percentile(sorted_latencies, 95),
                "min": min(self.all_latencies),
                "max": max(self.all_latencies),
            },
            "retrieval_time": {
                "mean": sum(self.all_retrieval_times) / len(self.all_retrieval_times),
                "p50": percentile(sorted_retrieval, 50),
                "p95": percentile(sorted_retrieval, 95),
            },
            "generation_time": {
                "mean": sum(self.all_generation_times) / len(self.all_generation_times),
                "p50": percentile(sorted_generation, 50),
                "p95": percentile(sorted_generation, 95),
            },
            "quality": {
                "mean": sum(recent_quality) / len(recent_quality),
                "min": min(recent_quality),
                "max": max(recent_quality),
            },
        }
        
        # Strategy distribution
        if recent:
            strategy_counts = {}
            for m in recent:
                strategy_counts[m.strategy_used] = strategy_counts.get(m.strategy_used, 0) + 1
            stats["strategy_distribution"] = strategy_counts
            
            # Average K used
            stats["avg_k_used"] = sum(m.top_k_used for m in recent) / len(recent)
        
        return stats
    
    def _save_history(self, path: str):
        """Persist history to a JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "total_queries": self.total_queries,
            "history": [asdict(m) for m in self.history],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_history(self, path: str):
        """Load persisted history from JSON."""
        filepath = Path(path)
        if not filepath.exists():
            return
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.total_queries = data.get("total_queries", 0)
            for item in data.get("history", []):
                self.history.append(QueryMetrics(**item))
                self.all_latencies.append(item["total_latency"])
                self.all_retrieval_times.append(item["retrieval_time"])
                self.all_generation_times.append(item["generation_time"])
        except Exception as e:
            print(f"  [WARNING] Could not load feedback history: {e}")
