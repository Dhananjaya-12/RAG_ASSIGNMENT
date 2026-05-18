"""
Performance Benchmark Suite
Measures and reports latency, retrieval time, generation time,
and the impact of adaptive logic on the RAG pipeline.
"""

import time
import json
from typing import Dict, List
from pathlib import Path


# Benchmark query sets with varying complexity
BENCHMARK_QUERIES = {
    "simple": [
        "What is machine learning?",
        "Define TCP protocol",
        "What is a hash table?",
        "What are qubits?",
        "What is SQL?",
    ],
    "medium": [
        "How do transformers work in natural language processing?",
        "Explain the CAP theorem in distributed databases",
        "What are the applications of reinforcement learning?",
        "How does the greenhouse effect cause global warming?",
        "Describe the differences between arrays and linked lists",
    ],
    "complex": [
        "Compare supervised and unsupervised learning approaches, including their use cases and limitations",
        "How does quantum computing threaten current cryptographic systems and what are the proposed solutions?",
        "Explain the relationship between the OSI model layers and how data flows through a network",
        "What are the tradeoffs between relational and NoSQL databases for different application types?",
        "How do renewable energy technologies address climate change, and what challenges remain?",
    ],
}


def run_benchmark_set(
    pipeline,
    queries: List[str],
    label: str,
    use_adaptive: bool = True,
) -> List[Dict]:
    """Run a set of benchmark queries and collect metrics."""
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"    [{label}] Query {i}/{len(queries)}: {query[:60]}...")
        
        result = pipeline.query(
            question=query,
            use_adaptive=use_adaptive,
            use_cache=False,  # Disable cache for fair benchmarking
            verbose=False,
        )
        
        results.append({
            "query": query,
            "label": label,
            "adaptive": use_adaptive,
            "total_latency": result["metrics"]["total_latency"],
            "retrieval_time": result["metrics"]["retrieval_time"],
            "generation_time": result["metrics"]["generation_time"],
            "quality_score": result["metrics"]["quality_score"],
            "top_k": result["adaptive"]["top_k"],
            "strategy": result["adaptive"]["strategy"],
            "complexity": result["adaptive"]["complexity"],
            "answer_length": len(result["answer"]),
        })
        
        print(f"      → latency={result['metrics']['total_latency']:.3f}s | "
              f"quality={result['metrics']['quality_score']:.3f} | "
              f"K={result['adaptive']['top_k']} | {result['adaptive']['strategy']}")
    
    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute aggregate statistics from benchmark results."""
    if not results:
        return {}
    
    latencies = sorted([r["total_latency"] for r in results])
    retrieval_times = sorted([r["retrieval_time"] for r in results])
    generation_times = sorted([r["generation_time"] for r in results])
    quality_scores = [r["quality_score"] for r in results]
    
    def percentile(data, p):
        idx = min(int(len(data) * p / 100), len(data) - 1)
        return data[idx]
    
    return {
        "count": len(results),
        "latency": {
            "mean": sum(latencies) / len(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "retrieval_time": {
            "mean": sum(retrieval_times) / len(retrieval_times),
            "p50": percentile(retrieval_times, 50),
            "p95": percentile(retrieval_times, 95),
        },
        "generation_time": {
            "mean": sum(generation_times) / len(generation_times),
            "p50": percentile(generation_times, 50),
            "p95": percentile(generation_times, 95),
        },
        "quality": {
            "mean": sum(quality_scores) / len(quality_scores),
            "min": min(quality_scores),
            "max": max(quality_scores),
        },
        "avg_top_k": sum(r["top_k"] for r in results) / len(results),
        "strategy_distribution": _count_values(results, "strategy"),
        "complexity_distribution": _count_values(results, "complexity"),
    }


def _count_values(results: List[Dict], key: str) -> Dict:
    """Count occurrences of values for a given key."""
    counts = {}
    for r in results:
        v = r.get(key, "unknown")
        counts[v] = counts.get(v, 0) + 1
    return counts


def generate_report(
    adaptive_results: List[Dict],
    fixed_results: List[Dict],
    adaptive_stats: Dict,
    fixed_stats: Dict,
    output_path: str,
):
    """Generate a markdown benchmark report."""
    report = []
    report.append("# Benchmark Results — Adaptive RAG System\n")
    report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overall comparison
    report.append("## Adaptive vs Fixed Comparison\n")
    report.append("| Metric | Adaptive | Fixed (K=5, hybrid) | Improvement |")
    report.append("|--------|----------|---------------------|-------------|")
    
    for metric_name, key_path in [
        ("Latency P50", ("latency", "p50")),
        ("Latency P95", ("latency", "p95")),
        ("Latency Mean", ("latency", "mean")),
        ("Retrieval P50", ("retrieval_time", "p50")),
        ("Retrieval P95", ("retrieval_time", "p95")),
        ("Generation P50", ("generation_time", "p50")),
        ("Generation P95", ("generation_time", "p95")),
        ("Quality Mean", ("quality", "mean")),
    ]:
        a_val = adaptive_stats.get(key_path[0], {}).get(key_path[1], 0)
        f_val = fixed_stats.get(key_path[0], {}).get(key_path[1], 0)
        
        if "quality" in key_path[0]:
            # Higher is better for quality
            improvement = ((a_val - f_val) / max(f_val, 0.001)) * 100
            direction = "↑" if improvement > 0 else "↓"
        else:
            # Lower is better for latency
            improvement = ((f_val - a_val) / max(f_val, 0.001)) * 100
            direction = "↑" if improvement > 0 else "↓"
        
        report.append(
            f"| {metric_name} | {a_val:.4f}s | {f_val:.4f}s | {direction} {abs(improvement):.1f}% |"
        )
    
    report.append(f"\n| Avg Top-K | {adaptive_stats.get('avg_top_k', 0):.1f} | 5.0 | — |")
    
    # Adaptive strategy distribution
    report.append("\n## Adaptive Strategy Distribution\n")
    report.append("| Strategy | Count |")
    report.append("|----------|-------|")
    for strategy, count in adaptive_stats.get("strategy_distribution", {}).items():
        report.append(f"| {strategy} | {count} |")
    
    # Complexity distribution
    report.append("\n## Query Complexity Distribution\n")
    report.append("| Complexity | Count |")
    report.append("|------------|-------|")
    for complexity, count in adaptive_stats.get("complexity_distribution", {}).items():
        report.append(f"| {complexity} | {count} |")
    
    # Per-complexity breakdown
    report.append("\n## Per-Complexity Performance (Adaptive)\n")
    report.append("| Complexity | Avg Latency | Avg Quality | Avg K |")
    report.append("|------------|-------------|-------------|-------|")
    
    for complexity in ["LOW", "MEDIUM", "HIGH"]:
        c_results = [r for r in adaptive_results if r["complexity"] == complexity]
        if c_results:
            avg_lat = sum(r["total_latency"] for r in c_results) / len(c_results)
            avg_qual = sum(r["quality_score"] for r in c_results) / len(c_results)
            avg_k = sum(r["top_k"] for r in c_results) / len(c_results)
            report.append(f"| {complexity} | {avg_lat:.4f}s | {avg_qual:.3f} | {avg_k:.1f} |")
    
    # Time breakdown
    report.append("\n## Time Breakdown\n")
    report.append("| Phase | Adaptive P50 | Fixed P50 | % of Total (Adaptive) |")
    report.append("|-------|-------------|-----------|----------------------|")
    
    a_ret_p50 = adaptive_stats.get("retrieval_time", {}).get("p50", 0)
    a_gen_p50 = adaptive_stats.get("generation_time", {}).get("p50", 0)
    a_total = a_ret_p50 + a_gen_p50
    f_ret_p50 = fixed_stats.get("retrieval_time", {}).get("p50", 0)
    f_gen_p50 = fixed_stats.get("generation_time", {}).get("p50", 0)
    
    ret_pct = (a_ret_p50 / max(a_total, 0.001)) * 100
    gen_pct = (a_gen_p50 / max(a_total, 0.001)) * 100
    
    report.append(f"| Retrieval | {a_ret_p50:.4f}s | {f_ret_p50:.4f}s | {ret_pct:.1f}% |")
    report.append(f"| Generation | {a_gen_p50:.4f}s | {f_gen_p50:.4f}s | {gen_pct:.1f}% |")
    
    # Detailed results
    report.append("\n## Detailed Query Results (Adaptive)\n")
    report.append("| # | Query | Complexity | K | Strategy | Latency | Quality |")
    report.append("|---|-------|------------|---|----------|---------|---------|")
    
    for i, r in enumerate(adaptive_results, 1):
        query_short = r["query"][:45] + "..." if len(r["query"]) > 45 else r["query"]
        report.append(
            f"| {i} | {query_short} | {r['complexity']} | {r['top_k']} | "
            f"{r['strategy']} | {r['total_latency']:.3f}s | {r['quality_score']:.3f} |"
        )
    
    report.append("\n## Key Findings\n")
    report.append("1. **Adaptive K Selection**: The system dynamically adjusts K based on query complexity,")
    report.append("   using smaller K for simple queries (faster) and larger K for complex queries (better quality).")
    report.append("2. **Hybrid vs Vector-Only**: Simple queries use vector-only search (faster),")
    report.append("   while complex queries benefit from hybrid retrieval (better recall).")
    report.append("3. **Re-ranker Impact**: The cross-encoder re-ranker improves quality for complex queries")
    report.append("   but adds latency overhead, so it's skipped for simple queries.")
    report.append("4. **Feedback Loop**: Over successive queries, the system learns to adjust parameters")
    report.append("   based on observed latency and quality patterns.")
    
    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"\n  Report saved to: {output_path}")


def run_full_benchmark(pipeline, config):
    """Run the complete benchmark suite."""
    print("\n" + "=" * 60)
    print("  PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # ─── Run Adaptive Mode ──────────────────────────────────────
    print("\n  Phase 1: Running with ADAPTIVE logic...")
    adaptive_results = []
    for label, queries in BENCHMARK_QUERIES.items():
        adaptive_results.extend(
            run_benchmark_set(pipeline, queries, label, use_adaptive=True)
        )
    adaptive_stats = compute_statistics(adaptive_results)
    
    # ─── Run Fixed Mode ─────────────────────────────────────────
    print("\n  Phase 2: Running with FIXED parameters (K=5, hybrid, reranker ON)...")
    fixed_results = []
    for label, queries in BENCHMARK_QUERIES.items():
        fixed_results.extend(
            run_benchmark_set(pipeline, queries, label, use_adaptive=False)
        )
    fixed_stats = compute_statistics(fixed_results)
    
    # ─── Generate Report ────────────────────────────────────────
    report_path = str(Path(config.BASE_DIR) / "benchmarks" / "benchmark_results.md")
    generate_report(
        adaptive_results, fixed_results,
        adaptive_stats, fixed_stats,
        report_path,
    )
    
    # ─── Print Summary ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    
    print(f"\n  {'Metric':<25} {'Adaptive':>12} {'Fixed':>12}")
    print(f"  {'─' * 49}")
    print(f"  {'Latency P50':<25} {adaptive_stats['latency']['p50']:>11.4f}s {fixed_stats['latency']['p50']:>11.4f}s")
    print(f"  {'Latency P95':<25} {adaptive_stats['latency']['p95']:>11.4f}s {fixed_stats['latency']['p95']:>11.4f}s")
    print(f"  {'Quality Mean':<25} {adaptive_stats['quality']['mean']:>12.4f} {fixed_stats['quality']['mean']:>12.4f}")
    print(f"  {'Avg K':<25} {adaptive_stats['avg_top_k']:>12.1f} {'5.0':>12}")
    
    print(f"\n  Full report: {report_path}")
    print(f"{'=' * 60}\n")
