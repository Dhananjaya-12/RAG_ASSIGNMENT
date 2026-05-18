"""
Query Decomposition Module (Bonus)
Detects multi-part queries and splits them into sub-queries
for more focused retrieval.
"""

import re
from typing import List, Tuple


class QueryDecomposer:
    """
    Decomposes complex, multi-part queries into simpler sub-queries.
    
    Strategies:
    1. Split on conjunctions ("and", "also", "additionally")
    2. Split on multiple question marks
    3. Split on semicolons and "plus" patterns
    4. Detect comparison queries and create sub-queries per entity
    """
    
    # Patterns that indicate a query should be decomposed
    SPLIT_PATTERNS = [
        r'\?\s*(?:and|also|additionally)\s+',  # "? and what about..."
        r'\?\s+\w',                              # Multiple questions
        r'\s+(?:and also|and additionally)\s+',   # "X and also Y"
        r';\s+',                                  # Semicolon-separated
    ]
    
    CONJUNCTION_PATTERNS = [
        r'\s+and\s+(?:what|how|why|when|where|who)\s+',  # "X and what Y"
        r'\s+(?:also|additionally|furthermore|moreover)\s+',
    ]
    
    def should_decompose(self, query: str) -> bool:
        """
        Determine if a query should be decomposed.
        
        Returns True if the query appears to contain multiple sub-queries.
        """
        # Multiple question marks
        if query.count('?') > 1:
            return True
        
        # Contains conjunction patterns
        for pattern in self.CONJUNCTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Very long query (likely multi-part)
        if len(query.split()) > 20:
            return True
        
        # Contains comparison keywords with multiple entities
        if self._is_comparison_query(query):
            return True
        
        return False
    
    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        
        Args:
            query: The original complex query
        
        Returns:
            List of sub-queries. Returns [query] if decomposition isn't needed.
        """
        if not self.should_decompose(query):
            return [query]
        
        sub_queries = []
        
        # Strategy 1: Split on multiple question marks
        if query.count('?') > 1:
            parts = [p.strip() + '?' for p in query.split('?') if p.strip()]
            sub_queries.extend(parts)
        
        # Strategy 2: Split on conjunction patterns
        elif any(re.search(p, query, re.IGNORECASE) for p in self.CONJUNCTION_PATTERNS):
            # Try splitting on "and what/how/why..."
            parts = re.split(
                r'\s+(?:and\s+)?(?:also|additionally|furthermore|moreover)\s+',
                query,
                flags=re.IGNORECASE
            )
            if len(parts) < 2:
                # Try splitting on "and" before question words
                parts = re.split(
                    r'\s+and\s+(?=(?:what|how|why|when|where|who)\s)',
                    query,
                    flags=re.IGNORECASE
                )
            sub_queries.extend([p.strip() for p in parts if p.strip()])
        
        # Strategy 3: Handle comparison queries
        elif self._is_comparison_query(query):
            entities = self._extract_comparison_entities(query)
            if len(entities) >= 2:
                for entity in entities:
                    sub_queries.append(f"What is {entity}?")
                sub_queries.append(query)  # Also include original for comparison context
        
        # Strategy 4: Long query — try sentence splitting
        elif len(query.split()) > 20:
            sentences = re.split(r'[.;]\s+', query)
            sub_queries.extend([s.strip() for s in sentences if len(s.strip()) > 10])
        
        # Fallback: return original query
        if not sub_queries:
            return [query]
        
        # Clean and deduplicate
        cleaned = []
        seen = set()
        for sq in sub_queries:
            sq = sq.strip().rstrip('.')
            if sq and sq.lower() not in seen:
                seen.add(sq.lower())
                cleaned.append(sq)
        
        return cleaned if cleaned else [query]
    
    def merge_results(self, results_per_query: List[List[dict]]) -> List[dict]:
        """
        Merge results from multiple sub-queries, removing duplicates
        and prioritizing documents that appear in multiple result sets.
        
        Args:
            results_per_query: List of result lists, one per sub-query
        
        Returns:
            Merged and deduplicated list of results
        """
        # Track document indices and their cumulative scores
        doc_scores = {}
        doc_data = {}
        
        for results in results_per_query:
            for result in results:
                idx = result.get("index", id(result))
                if idx in doc_scores:
                    # Boost score for documents appearing in multiple sub-query results
                    doc_scores[idx] += result.get("score", 0) * 0.8
                else:
                    doc_scores[idx] = result.get("score", 0)
                    doc_data[idx] = result
        
        # Sort by cumulative score
        sorted_indices = sorted(doc_scores.keys(), key=lambda i: doc_scores[i], reverse=True)
        
        merged = []
        for idx in sorted_indices:
            result = doc_data[idx].copy()
            result["score"] = doc_scores[idx]
            result["multi_query_boost"] = idx in doc_scores
            merged.append(result)
        
        return merged
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if this is a comparison query."""
        comparison_keywords = [
            "compare", "contrast", "difference", "differences",
            "versus", "vs", "vs.", "compared to", "differ",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in comparison_keywords)
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """Extract entities being compared from a comparison query."""
        query_lower = query.lower()
        
        # Pattern: "X vs Y", "X versus Y", "X compared to Y"
        patterns = [
            r'(?:compare|contrast)\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+?)(?:\?|$)',
            r'(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)',
            r'difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        return []
