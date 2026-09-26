#!/usr/bin/env python3
"""
EXP-0002: Blocking & Candidate Generation Evaluation Suite
Amazon ML Challenge 2026 — Business Entity Resolution

Compares multiple candidate generation strategies against Ground Truth:
- Strategy 1: exact_name
- Strategy 2: rare_tokens
- Strategy 3: disjunctive_union
- Strategy 4: composite_union

Measures:
- True-pair candidate recall (overall, S2, S3)
- S1-entity level coverage
- Candidate explosion / volume statistics
- Throughput (QPS) and memory usage
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Setup sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import get_dataset_dir, BASE_DIR, EXPERIMENTS_DIR, SEED
    from src.data_loading import load_data_list, load_data_generator
    from src.normalization import normalize_record
    from src.blocking.token_index import InvertedTokenIndex
    from src.blocking.strategies import STRATEGIES
    from src.blocking.evaluator import load_ground_truth_map, evaluate_blocking_strategy, get_process_memory_mb
except ImportError:
    from code.business_entity_resolution.src.config import get_dataset_dir, BASE_DIR, EXPERIMENTS_DIR, SEED
    from code.business_entity_resolution.src.data_loading import load_data_list, load_data_generator
    from code.business_entity_resolution.src.normalization import normalize_record
    from code.business_entity_resolution.src.blocking.token_index import InvertedTokenIndex
    from code.business_entity_resolution.src.blocking.strategies import STRATEGIES
    from code.business_entity_resolution.src.blocking.evaluator import load_ground_truth_map, evaluate_blocking_strategy, get_process_memory_mb


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def run_blocking_experiment(
    data_dir=None,
    s1_sample_size=2000,
    target_sample_size=100000,
    top_k=50,
    max_token_freq=5000,
    strategies_to_test=None,
    experiment_id="EXP-0002_blocking_eval",
    seed=42,
    save_json=True
):
    print("=" * 80)
    print(f"AMAZON ML CHALLENGE 2026 — {experiment_id}")
    print("BLOCKING & CANDIDATE GENERATION EVALUATION")
    print("=" * 80)
    
    random.seed(seed)
    resolved_data_dir = get_dataset_dir(data_dir)
    git_commit = get_git_commit()
    
    train_dir = resolved_data_dir / "train"
    gt_file = train_dir / "train_ground_truth.tsv"
    s1_file = train_dir / "train_source1.tsv"
    s2_file = train_dir / "train_source2.tsv"
    s3_file = train_dir / "train_source3.tsv"
    
    assert gt_file.exists(), f"Ground truth file missing: {gt_file}"
    assert s1_file.exists(), f"Source1 file missing: {s1_file}"
    assert s2_file.exists(), f"Source2 file missing: {s2_file}"
    assert s3_file.exists(), f"Source3 file missing: {s3_file}"
    
    print(f"\n[1] Configuration & Setup:")
    print(f"    Git Commit:         {git_commit}")
    print(f"    Dataset Dir:        {resolved_data_dir}")
    print(f"    S1 Query Sample:    {s1_sample_size if s1_sample_size else 'ALL'}")
    print(f"    Target S2/S3 Sample:{target_sample_size if target_sample_size else 'ALL'}")
    print(f"    Candidate Top-K:    {top_k}")
    print(f"    Max Token Freq:     {max_token_freq}")
    print(f"    Initial Memory:     {get_process_memory_mb()} MB")

    # 1. Load S1 Query records
    print(f"\n[2] Loading Query Records (Source1)...")
    t0 = time.time()
    s1_records = load_data_list(s1_file, limit=s1_sample_size)
    s1_ids = {r["entity_id"] for r in s1_records}
    print(f"    Loaded {len(s1_records):,} S1 queries in {time.time() - t0:.2f}s")
    
    # 2. Load Ground Truth for S1 subset
    print(f"\n[3] Loading Ground Truth Labels...")
    t1 = time.time()
    gt_map = load_ground_truth_map(gt_file, s1_filter_ids=s1_ids)
    total_gt_matches = sum(len(m) for m in gt_map.values())
    print(f"    Mapped {len(gt_map):,} S1 entities to {total_gt_matches:,} true match pairs in {time.time() - t1:.2f}s")

    # 3. Build Inverted Indexes for S2 and S3
    print(f"\n[4] Building Target Inverted Indexes (S2 & S3)...")
    t_idx0 = time.time()
    s2_index = InvertedTokenIndex(max_token_freq=max_token_freq)
    s3_index = InvertedTokenIndex(max_token_freq=max_token_freq)
    
    s2_count = 0
    for row in load_data_generator(s2_file):
        s2_index.add_record(row)
        s2_count += 1
        if target_sample_size and s2_count >= target_sample_size:
            break
            
    s3_count = 0
    for row in load_data_generator(s3_file):
        s3_index.add_record(row)
        s3_count += 1
        if target_sample_size and s3_count >= target_sample_size:
            break
            
    index_time = round(time.time() - t_idx0, 2)
    print(f"    Indexed {s2_count:,} S2 records + {s3_count:,} S3 records in {index_time}s")
    print(f"    S2 Unique Tokens: {len(s2_index.global_token_index):,} | S3 Unique Tokens: {len(s3_index.global_token_index):,}")
    print(f"    Memory after indexing: {get_process_memory_mb()} MB")

    # 4. Evaluate Strategies
    if not strategies_to_test:
        strategies_to_test = list(STRATEGIES.keys())
        
    print(f"\n[5] Benchmarking Candidate Generation Strategies...")
    results = {}
    
    for strat_name in strategies_to_test:
        strat_fn = STRATEGIES[strat_name]
        print(f"    -> Evaluating: {strat_name} ...")
        eval_metrics = evaluate_blocking_strategy(
            strat_fn,
            s1_records,
            s2_index,
            s3_index,
            gt_map,
            top_k=top_k
        )
        results[strat_name] = eval_metrics

    # 5. Comparative Summary Table
    print("\n" + "=" * 105)
    print("STRATEGY COMPARISON SUMMARY TABLE")
    print("=" * 105)
    header = f"{'Strategy':<20} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'S1 Cov':<8} | {'Mean Cand':<9} | {'P95 Cand':<8} | {'QPS':<8} | {'Time (s)'}"
    print(header)
    print("-" * 105)
    
    for strat_name, m in results.items():
        rec = f"{m['true_pair_recall']*100:.2f}%"
        s2_r = f"{m['s2_recall']*100:.2f}%"
        s3_r = f"{m['s3_recall']*100:.2f}%"
        cov = f"{m['s1_entity_coverage']*100:.2f}%"
        mean_c = f"{m['candidate_volume']['mean_per_s1']:.1f}"
        p95_c = f"{m['candidate_volume']['p95_per_s1']:.0f}"
        qps = f"{m['performance']['queries_per_sec']:.1f}"
        t_sec = f"{m['performance']['query_time_sec']:.2f}"
        print(f"{strat_name:<20} | {rec:<8} | {s2_r:<8} | {s3_r:<8} | {cov:<8} | {mean_c:<9} | {p95_c:<8} | {qps:<8} | {t_sec}")
    print("=" * 105)

    # 6. Save Telemetry
    experiment_payload = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "configuration": {
            "s1_sample_size": s1_sample_size,
            "target_sample_size": target_sample_size,
            "top_k": top_k,
            "max_token_freq": max_token_freq,
            "seed": seed
        },
        "target_index_stats": {
            "s2_records_indexed": s2_count,
            "s3_records_indexed": s3_count,
            "index_build_time_sec": index_time,
            "memory_post_index_mb": get_process_memory_mb()
        },
        "strategy_results": results
    }
    
    if save_json:
        exp_dir = EXPERIMENTS_DIR / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        res_file = exp_dir / "results.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(experiment_payload, f, indent=2)
            
        meta_file = exp_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(experiment_payload, f, indent=2)
            
        print(f"\n[OK] Experiment results saved to: {res_file}")

    return experiment_payload


def main():
    parser = argparse.ArgumentParser(description="EXP-0002 Blocking Strategy Evaluation")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--sample-size", type=int, default=1000, help="S1 query sample size (e.g. 1000, 5000, 10000)")
    parser.add_argument("--target-sample-size", type=int, default=50000, help="Target S2/S3 sample size per source")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate capacity per query")
    parser.add_argument("--max-token-freq", type=int, default=5000, help="Frequency limit for inverted index tokens")
    parser.add_argument("--strategies", nargs="+", default=None, help="Strategies to evaluate")
    parser.add_argument("--experiment-id", type=str, default="EXP-0002_blocking_eval", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    
    args = parser.parse_args()
    
    run_blocking_experiment(
        data_dir=args.data_dir,
        s1_sample_size=args.sample_size,
        target_sample_size=args.target_sample_size,
        top_k=args.top_k,
        max_token_freq=args.max_token_freq,
        strategies_to_test=args.strategies,
        experiment_id=args.experiment_id,
        seed=args.seed,
        save_json=True
    )


if __name__ == "__main__":
    main()
