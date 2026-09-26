#!/usr/bin/env python3
"""
EXP-0002: Corrected Blocking & Candidate Generation Evaluation Suite
Amazon ML Challenge 2026 — Business Entity Resolution

Methodology:
1. Sample S1 query entities deterministically.
2. Extract ALL true positive S2 & S3 target IDs belonging to the sampled S1 queries from Ground Truth.
3. Construct a controlled retrieval universe by streaming S2 and S3:
   - 100% inclusion of all true positive target records.
   - Controlled negative distractor records up to target_sample_size.
4. Strictly assert universe completeness before evaluation.
5. Compute true candidate recall over ground truth pairs present in the retrieval universe.
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
from collections import defaultdict

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


def index_target_source_with_guaranteed_positives(
    source_file: Path,
    required_positive_ids: set,
    target_sample_size: int,
    max_token_freq: int = 5000,
    source_name: str = "Target"
):
    """
    Streams a target source TSV (e.g. S2 or S3) to guarantee 100% inclusion of all
    required true positive targets for the sampled S1 queries, while filling the rest
    of the index with negative distractor records up to target_sample_size.
    """
    import csv
    csv.field_size_limit(sys.maxsize)
    
    index = InvertedTokenIndex(max_token_freq=max_token_freq)
    indexed_ids = set()
    
    found_positives = 0
    total_positives = len(required_positive_ids)
    negative_quota = max(0, target_sample_size - total_positives) if target_sample_size else float("inf")
    negative_count = 0
    
    t0 = time.time()
    with open(source_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id")
            if not eid:
                continue
                
            is_pos = eid in required_positive_ids
            is_neg = (not is_pos) and (negative_count < negative_quota)
            
            if is_pos or is_neg:
                norm_row = normalize_record(row)
                index.add_record(norm_row)
                indexed_ids.add(eid)
                if is_pos:
                    found_positives += 1
                else:
                    negative_count += 1
                
            # Early exit only if ALL positive targets have been found AND negative quota is met
            if target_sample_size and found_positives >= total_positives and negative_count >= negative_quota:
                break
            
    elapsed = round(time.time() - t0, 2)
    
    # Strict assertion: every required positive target MUST be present in the index
    missing_positives = required_positive_ids - indexed_ids
    assert len(missing_positives) == 0, (
        f"[FATAL AUDIT ERROR] {len(missing_positives)} true positive {source_name} target records were missing "
        f"from {source_file.name}! Universe completeness invariant violated."
    )
    
    return index, indexed_ids, found_positives, negative_count, elapsed



def run_blocking_experiment(
    data_dir=None,
    s1_sample_size=1000,
    target_sample_size=50000,
    top_k=50,
    max_token_freq=5000,
    strategies_to_test=None,
    experiment_id="EXP-0002_blocking_eval",
    seed=42,
    save_json=True
):
    print("=" * 85)
    print(f"AMAZON ML CHALLENGE 2026 — {experiment_id}")
    print("CORRECTED BLOCKING & CANDIDATE GENERATION EVALUATION")
    print("=" * 85)
    
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
    
    # 1. Load S1 Query records deterministically
    print(f"\n[1] Loading Deterministic S1 Query Records (Source1)...")
    t0 = time.time()
    s1_records = load_data_list(s1_file, limit=s1_sample_size)
    s1_ids = {r["entity_id"] for r in s1_records}
    print(f"    Loaded {len(s1_records):,} S1 queries in {time.time() - t0:.2f}s")
    
    # 2. Extract Ground Truth mapping and target requirements
    print(f"\n[2] Extracting Ground Truth Target Mappings...")
    t1 = time.time()
    gt_map = load_ground_truth_map(gt_file, s1_filter_ids=s1_ids)
    
    required_s2_ids = {m for matches in gt_map.values() for m in matches if m.startswith("S2-")}
    required_s3_ids = {m for matches in gt_map.values() for m in matches if m.startswith("S3-")}
    total_true_pairs = sum(len(m) for m in gt_map.values())
    
    print(f"    S1 Entities with Ground Truth: {len(gt_map):,} / {len(s1_records):,}")
    print(f"    Required S2 Positive Targets:  {len(required_s2_ids):,}")
    print(f"    Required S3 Positive Targets:  {len(required_s3_ids):,}")
    print(f"    Total Ground Truth True Pairs: {total_true_pairs:,} in {time.time() - t1:.2f}s")

    # 3. Construct Guaranteed Retrieval Universe for S2 and S3
    print(f"\n[3] Building Retrieval Universe (Positives Guaranteed + Distractor Sampling)...")
    
    print(f"    -> Streaming S2: guaranteeing {len(required_s2_ids):,} positives + up to {target_sample_size:,} distractors...")
    s2_index, indexed_s2_ids, pos_s2, neg_s2, time_s2 = index_target_source_with_guaranteed_positives(
        s2_file, required_s2_ids, target_sample_size, max_token_freq=max_token_freq, source_name="S2"
    )
    print(f"       [OK] S2 Universe: {len(indexed_s2_ids):,} records ({pos_s2:,} positive + {neg_s2:,} negative) indexed in {time_s2}s")
    
    print(f"    -> Streaming S3: guaranteeing {len(required_s3_ids):,} positives + up to {target_sample_size:,} distractors...")
    s3_index, indexed_s3_ids, pos_s3, neg_s3, time_s3 = index_target_source_with_guaranteed_positives(
        s3_file, required_s3_ids, target_sample_size, max_token_freq=max_token_freq, source_name="S3"
    )
    print(f"       [OK] S3 Universe: {len(indexed_s3_ids):,} records ({pos_s3:,} positive + {neg_s3:,} negative) indexed in {time_s3}s")
    
    indexed_target_ids = indexed_s2_ids | indexed_s3_ids
    total_universe_size = len(indexed_target_ids)
    print(f"    Total Target Retrieval Universe: {total_universe_size:,} records | Memory RSS: {get_process_memory_mb()} MB")

    # 4. Evaluate Strategies
    if not strategies_to_test:
        strategies_to_test = list(STRATEGIES.keys())
        
    print(f"\n[4] Benchmarking Candidate Generation Strategies on Validated Universe...")
    results = {}
    
    for strat_name in strategies_to_test:
        strat_fn = STRATEGIES[strat_name]
        print(f"    -> Evaluating: {strat_name} (top_k={top_k}) ...")
        eval_metrics = evaluate_blocking_strategy(
            strat_fn,
            s1_records,
            s2_index,
            s3_index,
            gt_map,
            indexed_target_ids=indexed_target_ids,
            top_k=top_k
        )
        results[strat_name] = eval_metrics

    # 5. Comparative Summary Table
    print("\n" + "=" * 115)
    print("CORRECTED STRATEGY COMPARISON SUMMARY TABLE (METHODOLOGICALLY VALIDATED)")
    print("=" * 115)
    header = f"{'Strategy':<20} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'S1 Cov':<8} | {'Mean Cand':<9} | {'P95 Cand':<8} | {'Max Cand':<8} | {'QPS':<8} | {'Time (s)'}"
    print(header)
    print("-" * 115)
    
    for strat_name, m in results.items():
        rec = f"{m['true_pair_recall']*100:.2f}%"
        s2_r = f"{m['s2_recall']*100:.2f}%"
        s3_r = f"{m['s3_recall']*100:.2f}%"
        cov = f"{m['s1_entity_coverage']*100:.2f}%"
        mean_c = f"{m['candidate_volume']['mean_per_s1']:.1f}"
        p95_c = f"{m['candidate_volume']['p95_per_s1']:.0f}"
        max_c = f"{m['candidate_volume']['max_per_s1']}"
        qps = f"{m['performance']['queries_per_sec']:.1f}"
        t_sec = f"{m['performance']['query_time_sec']:.2f}"
        print(f"{strat_name:<20} | {rec:<8} | {s2_r:<8} | {s3_r:<8} | {cov:<8} | {mean_c:<9} | {p95_c:<8} | {max_c:<8} | {qps:<8} | {t_sec}")
    print("=" * 115)

    # 6. Save Telemetry
    experiment_payload = {
        "experiment_id": experiment_id,
        "evaluation_methodology": "corrected_guaranteed_positive_universe",
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "configuration": {
            "s1_sample_size": len(s1_records),
            "target_sample_size_per_source": target_sample_size,
            "top_k": top_k,
            "max_token_freq": max_token_freq,
            "seed": seed
        },
        "retrieval_universe": {
            "s1_queries_evaluated": len(s1_records),
            "s1_entities_with_gt": len(gt_map),
            "total_gt_pairs_evaluated": total_true_pairs,
            "s2_guaranteed_positives": pos_s2,
            "s2_negative_distractors": neg_s2,
            "s2_total_universe": len(indexed_s2_ids),
            "s3_guaranteed_positives": pos_s3,
            "s3_negative_distractors": neg_s3,
            "s3_total_universe": len(indexed_s3_ids),
            "total_target_universe": total_universe_size,
            "s2_index_time_sec": time_s2,
            "s3_index_time_sec": time_s3,
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
            
        print(f"\n[OK] Corrected experiment results saved to: {res_file}")

    return experiment_payload


def main():
    parser = argparse.ArgumentParser(description="EXP-0002 Corrected Blocking Strategy Evaluation")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--sample-size", type=int, default=1000, help="S1 query sample size (e.g. 1000, 2000, 5000)")
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
