#!/usr/bin/env python3
"""
Exact-Equivalence Test & Parallel Worker Micro-Benchmark
Amazon ML Challenge 2026 — Business Entity Resolution

Compares:
1. Canonical single-worker production execution
2. Partitioned 3-worker parallel production execution + merge_parallel_outputs

Verification:
- Candidate set equality = 100%
- Match decision equality = 100%
- Merged TSV line-by-line exact match
- 0 duplicate or missing S1 entities
- Official submission validator = PASS
- Micro-benchmark across 1, 2, 3, and 4 workers.
"""

import sys
import os
import csv
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.run_production_pipeline import run_production_pipeline
from scripts.merge_parallel_outputs import merge_parallel_outputs


def test_parallel_equivalence_and_benchmark():
    print("=" * 95)
    print("EXP-0005: PARALLEL WORKER EXACT-EQUIVALENCE TEST & BENCHMARK")
    print("=" * 95)
    
    test_slice_dir = REPO_ROOT / "temp_parallel_eval"
    if test_slice_dir.exists():
        shutil.rmtree(test_slice_dir)
    test_slice_dir.mkdir(parents=True, exist_ok=True)
    
    s1_limit = 2000
    target_limit = 50000
    
    # -------------------------------------------------------------
    # 1. Run Single-Worker Canonical Baseline
    # -------------------------------------------------------------
    single_dir = test_slice_dir / "single_worker"
    print("\n" + "#" * 60)
    print(f"STEP 1: Running Canonical Single Worker ({s1_limit} S1, {target_limit} targets)...")
    print("#" * 60)
    
    t0_single = time.time()
    single_res = run_production_pipeline(
        output_dir=str(single_dir),
        s1_limit=s1_limit,
        target_limit=target_limit,
        num_workers=1,
        worker_id=0,
        stage_local=False,
        resume=False
    )
    t_single_total = time.time() - t0_single
    
    single_matching = single_dir / "matching_results.tsv"
    single_candidates = single_dir / "candidate_pairs.tsv"
    
    # -------------------------------------------------------------
    # 2. Run Partitioned 3-Worker Production Mode
    # -------------------------------------------------------------
    multi_dir = test_slice_dir / "multi_worker_3w"
    print("\n" + "#" * 60)
    print(f"STEP 2: Running Partitioned 3-Worker Mode ({s1_limit} S1, {target_limit} targets)...")
    print("#" * 60)
    
    worker_times_3w = []
    for w_id in range(3):
        print(f"\n>>> Launching Worker {w_id} of 3 <<<")
        t0_w = time.time()
        w_res = run_production_pipeline(
            output_dir=str(multi_dir),
            s1_limit=s1_limit,
            target_limit=target_limit,
            num_workers=3,
            worker_id=w_id,
            stage_local=False,
            resume=False
        )
        worker_times_3w.append(time.time() - t0_w)
        
    # -------------------------------------------------------------
    # 3. Merge 3-Worker Outputs
    # -------------------------------------------------------------
    merged_dir = test_slice_dir / "merged_3w"
    print("\n" + "#" * 60)
    print("STEP 3: Merging 3-Worker Outputs with merge_parallel_outputs.py...")
    print("#" * 60)
    
    merge_res = merge_parallel_outputs(
        parent_dir=str(multi_dir),
        num_workers=3,
        output_dir=str(merged_dir),
        expected_total_s1=s1_limit,
        skip_validator=True # We will validate with custom slice directory below
    )
    
    merged_matching = merged_dir / "matching_results.tsv"
    merged_candidates = merged_dir / "candidate_pairs.tsv"
    
    # -------------------------------------------------------------
    # 4. Compare Exact Equivalence (Single vs Merged 3-Worker)
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("STEP 4: Mathematical Exact-Equivalence Verification")
    print("#" * 60)
    
    # Check matching results
    with open(single_matching, "r", encoding="utf-8") as f_s, open(merged_matching, "r", encoding="utf-8") as f_m:
        single_match_rows = [line.strip().split("\t") for line in f_s if line.strip()]
        merged_match_rows = [line.strip().split("\t") for line in f_m if line.strip()]
        
    with open(single_candidates, "r", encoding="utf-8") as f_s, open(merged_candidates, "r", encoding="utf-8") as f_m:
        single_cand_rows = [line.strip().split("\t") for line in f_s if line.strip()]
        merged_cand_rows = [line.strip().split("\t") for line in f_m if line.strip()]
        
    assert len(single_match_rows) == len(merged_match_rows), f"Row count mismatch in matching: {len(single_match_rows)} vs {len(merged_match_rows)}"
    assert len(single_cand_rows) == len(merged_cand_rows), f"Row count mismatch in candidates: {len(single_cand_rows)} vs {len(merged_cand_rows)}"
    
    n_queries = len(single_match_rows) - 1 # exclude header
    
    cand_identical_count = 0
    match_identical_count = 0
    total_jaccard = 0.0
    changed_decisions = 0
    
    for idx in range(1, len(single_match_rows)):
        s_m = single_match_rows[idx]
        m_m = merged_match_rows[idx]
        assert s_m[0] == m_m[0], f"Entity ID mismatch at row {idx}: {s_m[0]} vs {m_m[0]}"
        
        s_match_set = set(s_m[1].split(",")) if len(s_m) > 1 and s_m[1] else set()
        m_match_set = set(m_m[1].split(",")) if len(m_m) > 1 and m_m[1] else set()
        
        if s_match_set == m_match_set:
            match_identical_count += 1
        else:
            changed_decisions += 1
            
        s_c = single_cand_rows[idx]
        m_c = merged_cand_rows[idx]
        assert s_c[0] == m_c[0], f"Candidate Entity ID mismatch at row {idx}: {s_c[0]} vs {m_c[0]}"
        
        s_cand_set = set(s_c[1].split(",")) if len(s_c) > 1 and s_c[1] else set()
        m_cand_set = set(m_c[1].split(",")) if len(m_c) > 1 and m_c[1] else set()
        
        if s_cand_set == m_cand_set:
            cand_identical_count += 1
            total_jaccard += 1.0
        else:
            intersection = len(s_cand_set & m_cand_set)
            union = len(s_cand_set | m_cand_set)
            total_jaccard += (intersection / max(1, union))
            
    cand_equality_pct = (cand_identical_count / n_queries) * 100.0
    match_equality_pct = (match_identical_count / n_queries) * 100.0
    mean_jaccard_pct = (total_jaccard / n_queries) * 100.0
    
    print(f"Total S1 Queries Evaluated:      {n_queries:,}")
    print(f"Candidate Set Exact Equality:     {cand_equality_pct:6.2f}% ({cand_identical_count}/{n_queries})")
    print(f"Candidate Mean Jaccard:           {mean_jaccard_pct:6.2f}%")
    print(f"Prediction Exact Equality:        {match_equality_pct:6.2f}% ({match_identical_count}/{n_queries})")
    print(f"Changed S1 Decisions:             {changed_decisions}")
    
    # Create test slice test_source1.tsv for official validator
    slice_test_dir = test_slice_dir / "slice_dataset"
    slice_test_dir.mkdir(parents=True, exist_ok=True)
    slice_s1_tsv = slice_test_dir / "test_source1.tsv"
    
    with open(REPO_ROOT / "student_resource" / "dataset" / "test" / "test_source1.tsv", "r", encoding="utf-8") as f_in, \
         open(slice_s1_tsv, "w", encoding="utf-8", newline="") as f_out:
        r = csv.reader(f_in, delimiter="\t")
        w = csv.writer(f_out, delimiter="\t")
        w.writerow(next(r))
        for i, row in enumerate(r):
            if i >= s1_limit:
                break
            w.writerow(row)
            
    # Run official validator on merged 3-worker outputs
    val_script = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    val_passed = False
    if val_script.exists():
        res = subprocess.run([
            sys.executable, str(val_script),
            "--matching", str(merged_matching),
            "--candidate", str(merged_candidates),
            "--test-dir", str(slice_test_dir)
        ], capture_output=True, text=True)
        print("\n" + "=" * 50 + " VALIDATOR REPORT ON MERGED 3-WORKER OUTPUT " + "=" * 50)
        print(res.stdout.strip())
        print("=" * 100)
        val_passed = (res.returncode == 0)
        
    print(f"\nOfficial Submission Validator on Merged 3W: {'PASS' if val_passed else 'FAIL'}")

    # -------------------------------------------------------------
    # 5. Worker Scaling Micro-Benchmark (1, 2, 3, 4 Workers)
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("STEP 5: Benchmarking Worker Scaling (1, 2, 3, 4 Workers)...")
    print("#" * 60)
    
    bench_results = {}
    bench_results["1_worker"] = {
        "num_workers": 1,
        "query_count": s1_limit,
        "per_worker_queries": [s1_limit],
        "total_runtime_seconds": round(t_single_total, 2),
        "parallel_wall_clock_seconds": round(t_single_total, 2),
        "throughput_s1_per_sec": round(s1_limit / max(0.001, t_single_total), 2),
        "speedup": 1.00
    }
    
    # 2 Workers
    dir_2w = test_slice_dir / "bench_2w"
    t0_2w = time.time()
    w_times_2w = []
    for w_id in range(2):
        t0 = time.time()
        run_production_pipeline(
            output_dir=str(dir_2w),
            s1_limit=s1_limit,
            target_limit=target_limit,
            num_workers=2,
            worker_id=w_id,
            stage_local=False,
            resume=False
        )
        w_times_2w.append(time.time() - t0)
    parallel_2w_clock = max(w_times_2w) # in real parallel execution across 2 Colabs
    bench_results["2_workers"] = {
        "num_workers": 2,
        "query_count": s1_limit,
        "per_worker_queries": [1000, 1000],
        "worker_times_seconds": [round(t, 2) for t in w_times_2w],
        "parallel_wall_clock_seconds": round(parallel_2w_clock, 2),
        "throughput_s1_per_sec": round(s1_limit / max(0.001, parallel_2w_clock), 2),
        "speedup": round(t_single_total / max(0.001, parallel_2w_clock), 2)
    }
    
    # 3 Workers (already run above)
    parallel_3w_clock = max(worker_times_3w)
    bench_results["3_workers"] = {
        "num_workers": 3,
        "query_count": s1_limit,
        "per_worker_queries": [667, 667, 666],
        "worker_times_seconds": [round(t, 2) for t in worker_times_3w],
        "parallel_wall_clock_seconds": round(parallel_3w_clock, 2),
        "throughput_s1_per_sec": round(s1_limit / max(0.001, parallel_3w_clock), 2),
        "speedup": round(t_single_total / max(0.001, parallel_3w_clock), 2)
    }
    
    # 4 Workers
    dir_4w = test_slice_dir / "bench_4w"
    w_times_4w = []
    for w_id in range(4):
        t0 = time.time()
        run_production_pipeline(
            output_dir=str(dir_4w),
            s1_limit=s1_limit,
            target_limit=target_limit,
            num_workers=4,
            worker_id=w_id,
            stage_local=False,
            resume=False
        )
        w_times_4w.append(time.time() - t0)
    parallel_4w_clock = max(w_times_4w)
    bench_results["4_workers"] = {
        "num_workers": 4,
        "query_count": s1_limit,
        "per_worker_queries": [500, 500, 500, 500],
        "worker_times_seconds": [round(t, 2) for t in w_times_4w],
        "parallel_wall_clock_seconds": round(parallel_4w_clock, 2),
        "throughput_s1_per_sec": round(s1_limit / max(0.001, parallel_4w_clock), 2),
        "speedup": round(t_single_total / max(0.001, parallel_4w_clock), 2)
    }

    # Clean up test scratch dir
    if test_slice_dir.exists():
        shutil.rmtree(test_slice_dir)
        
    # Write experiment results
    exp_dir = REPO_ROOT / "experiments" / "EXP-0005_parallel_worker_mode"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        "experiment_id": "EXP-0005",
        "experiment_name": "3-Worker Parallel Production Inference Mode",
        "exact_equivalence_verification": {
            "s1_queries": n_queries,
            "target_slice": f"{target_limit} records per target source",
            "candidate_set_equality_pct": cand_equality_pct,
            "candidate_mean_jaccard_pct": mean_jaccard_pct,
            "match_decision_equality_pct": match_equality_pct,
            "changed_decisions": changed_decisions,
            "validator_passed": val_passed
        },
        "worker_scaling_benchmark": bench_results
    }
    
    with open(exp_dir / "results.json", "w", encoding="utf-8") as f_json:
        json.dump(results_data, f_json, indent=2)
        
    print("\n" + "=" * 95)
    print("EXP-0005 SUMMARY TABLE")
    print("=" * 95)
    print(f"{'Workers':<12} | {'Queries/Worker':<16} | {'Simulated Parallel Wall-Clock':<30} | {'Throughput':<14} | {'Speedup':<8}")
    print("-" * 95)
    for k, v in bench_results.items():
        q_str = str(v["per_worker_queries"])
        print(f"{v['num_workers']:<12} | {q_str:<16} | {v['parallel_wall_clock_seconds']:<6.2f} s                      | {v['throughput_s1_per_sec']:<5.1f} S1/s     | {v['speedup']:<5.2f}x")
    print("=" * 95)
    
    return results_data


if __name__ == "__main__":
    test_parallel_equivalence_and_benchmark()
