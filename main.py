"""
Adaptive RAG Inference System — Main Entry Point
Provides CLI for document ingestion, querying, benchmarking, and interactive mode.
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def print_banner():
    """Print the application banner."""
    print()
    print("+" + "=" * 62 + "+")
    print("|           ADAPTIVE RAG INFERENCE SYSTEM                    |")
    print("|     Self-Optimizing Retrieval-Augmented Generation         |")
    print("+" + "=" * 62 + "+")
    print()


def cmd_ingest(args):
    """Ingest documents and build the index."""
    import config
    from src.pipeline import AdaptiveRAGPipeline
    
    pipeline = AdaptiveRAGPipeline(config)
    stats = pipeline.ingest(args.data_dir)
    
    print("\n  Ingestion Summary:")
    print(f"    Documents: {stats['documents_loaded']}")
    print(f"    Chunks:    {stats['chunks_created']}")
    print(f"    Sources:   {', '.join(stats['sources'])}")


def cmd_query(args):
    """Run a single query."""
    import config
    from src.pipeline import AdaptiveRAGPipeline
    
    pipeline = AdaptiveRAGPipeline(config)
    
    # Try loading existing index, otherwise ingest
    if not pipeline.load_index():
        print("  No existing index found. Running ingestion first...")
        pipeline.ingest()
    
    print(f"\n  Query: {args.question}")
    print("-" * 60)
    
    result = pipeline.query(
        question=args.question,
        use_adaptive=not args.no_adaptive,
        use_cache=not args.no_cache,
        verbose=True,
    )
    
    print("\n" + "-" * 60)
    print("  ANSWER:")
    print("-" * 60)
    print(f"\n{result['answer']}\n")
    
    print("-" * 60)
    print("  SOURCES:")
    for src in result["sources"]:
        print(f"    * {src['file']} (chunk #{src['chunk_index']}, score: {src['score']:.3f})")
    
    print(f"\n  METRICS:")
    m = result["metrics"]
    print(f"    Total latency:   {m['total_latency']:.3f}s")
    print(f"    Retrieval time:  {m['retrieval_time']:.3f}s")
    print(f"    Generation time: {m['generation_time']:.3f}s")
    print(f"    Quality score:   {m['quality_score']:.3f}")
    
    a = result["adaptive"]
    print(f"\n  ADAPTIVE DECISIONS:")
    print(f"    Complexity:  {a['complexity']} ({a['complexity_score']:.2f})")
    print(f"    Top-K:       {a['top_k']}")
    print(f"    Strategy:    {a['strategy']}")
    print(f"    Re-ranker:   {'ON' if a['use_reranker'] else 'OFF'}")


def cmd_interactive(args):
    """Run interactive query mode."""
    import config
    from src.pipeline import AdaptiveRAGPipeline
    
    pipeline = AdaptiveRAGPipeline(config)
    
    if not pipeline.load_index():
        print("  No existing index found. Running ingestion first...")
        pipeline.ingest()
    
    print("\n  Interactive mode. Type 'quit' to exit, 'stats' for stats.\n")
    
    query_count = 0
    while True:
        try:
            question = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        
        if question.lower() == "stats":
            _print_stats(pipeline)
            continue
        
        if question.lower() == "cache":
            if pipeline.cache:
                stats = pipeline.cache.get_stats()
                print(f"    Cache entries: {stats['entries']}/{stats['max_size']}")
                print(f"    Hit rate: {stats['hit_rate']:.1%} ({stats['hits']} hits, {stats['misses']} misses)")
            else:
                print("    Cache disabled")
            continue
        
        query_count += 1
        print(f"\n  --- Query #{query_count} ---")
        
        result = pipeline.query(
            question=question,
            use_adaptive=True,
            use_cache=True,
            verbose=True,
        )
        
        print(f"\n  Answer:\n  {'-' * 56}")
        # Format answer with indentation
        for line in result["answer"].split("\n"):
            print(f"  {line}")
        print(f"  {'-' * 56}")
        
        # Compact metrics
        m = result["metrics"]
        a = result["adaptive"]
        print(f"  [{a['complexity']}] K={a['top_k']} | {a['strategy']} | "
              f"latency={m['total_latency']:.2f}s | quality={m['quality_score']:.2f}")
        
        if result.get("cache_hit"):
            print(f"  (served from cache)")
        
        print()


def cmd_benchmark(args):
    """Run the benchmark suite."""
    import config
    from benchmarks.run_benchmarks import run_full_benchmark
    
    from src.pipeline import AdaptiveRAGPipeline
    
    pipeline = AdaptiveRAGPipeline(config)
    
    if not pipeline.load_index():
        print("  No existing index found. Running ingestion first...")
        pipeline.ingest()
    
    run_full_benchmark(pipeline, config)


def cmd_demo(args):
    """Run a quick demo with sample queries."""
    import config
    from src.pipeline import AdaptiveRAGPipeline
    
    pipeline = AdaptiveRAGPipeline(config)
    
    if not pipeline.load_index():
        print("  Running initial document ingestion...")
        pipeline.ingest()
    
    demo_queries = [
        # LOW complexity (short, simple)
        "What is machine learning?",
        "Define TCP",
        # MEDIUM complexity
        "How do transformers work in NLP?",
        "Explain the CAP theorem in distributed databases",
        # HIGH complexity (comparison, multi-part)
        "Compare supervised and unsupervised learning. What are the key differences and when should each be used?",
        "How does quantum computing differ from classical computing, and what are the implications for cryptography?",
    ]
    
    print("\n" + "=" * 60)
    print("  DEMO: Running sample queries with varying complexity")
    print("=" * 60)
    
    for i, q in enumerate(demo_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"  Demo Query {i}/{len(demo_queries)}: {q}")
        print(f"{'=' * 60}")
        
        result = pipeline.query(question=q, verbose=True)
        
        # Truncate answer for demo
        answer = result["answer"]
        if len(answer) > 300:
            answer = answer[:300] + "..."
        print(f"\n  Answer: {answer}\n")
        
        m = result["metrics"]
        a = result["adaptive"]
        print(f"  [{a['complexity']}] K={a['top_k']} | {a['strategy']} | "
              f"reranker={'ON' if a['use_reranker'] else 'OFF'} | "
              f"latency={m['total_latency']:.2f}s | quality={m['quality_score']:.2f}")
    
    # Print final stats
    print(f"\n{'=' * 60}")
    print("  DEMO COMPLETE - Pipeline Statistics")
    print(f"{'=' * 60}")
    _print_stats(pipeline)


def _print_stats(pipeline):
    """Print pipeline statistics."""
    stats = pipeline.get_stats()
    
    print(f"\n  Pipeline Statistics:")
    print(f"    Index size:     {stats['pipeline']['index_size']} vectors")
    print(f"    LLM provider:   {stats['pipeline']['llm_provider']}")
    
    fb = stats.get("feedback", {})
    if fb.get("total_queries", 0) > 0:
        print(f"\n  Performance ({fb['total_queries']} queries):")
        lat = fb["latency"]
        print(f"    Latency  - P50: {lat['p50']:.3f}s | P95: {lat['p95']:.3f}s | Mean: {lat['mean']:.3f}s")
        ret = fb["retrieval_time"]
        print(f"    Retrieval - P50: {ret['p50']:.3f}s | P95: {ret['p95']:.3f}s")
        gen = fb["generation_time"]
        print(f"    Generation - P50: {gen['p50']:.3f}s | P95: {gen['p95']:.3f}s")
        qual = fb["quality"]
        print(f"    Quality  - Mean: {qual['mean']:.3f} | Min: {qual['min']:.3f} | Max: {qual['max']:.3f}")
        
        if "strategy_distribution" in fb:
            print(f"    Strategies: {fb['strategy_distribution']}")
        if "avg_k_used" in fb:
            print(f"    Avg K used: {fb['avg_k_used']:.1f}")
    
    if "cache" in stats:
        c = stats["cache"]
        print(f"\n  Cache:")
        print(f"    Entries: {c['entries']}/{c['max_size']}")
        print(f"    Hit rate: {c['hit_rate']:.1%} ({c['hits']} hits, {c['misses']} misses)")


def main():
    """Main entry point."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Adaptive RAG Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents and build index")
    ingest_parser.add_argument("--data-dir", default=None, help="Path to document directory")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive logic")
    query_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Interactive query mode")
    
    # Benchmark
    subparsers.add_parser("benchmark", help="Run performance benchmarks")
    
    # Demo
    subparsers.add_parser("demo", help="Run demo with sample queries")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n  Quick start:")
        print("    python main.py ingest        # Index documents")
        print("    python main.py demo           # Run demo queries")
        print("    python main.py interactive    # Interactive mode")
        print("    python main.py query \"What is machine learning?\"")
        print("    python main.py benchmark      # Run benchmarks")
        return
    
    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "interactive": cmd_interactive,
        "benchmark": cmd_benchmark,
        "demo": cmd_demo,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
