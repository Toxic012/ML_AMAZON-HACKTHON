import csv
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np

# Increase csv field size limit safely
csv.field_size_limit(sys.maxsize)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_process_memory_mb() -> float:
    if HAS_PSUTIL:
        try:
            return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
        except Exception:
            pass
    return 0.0


def load_ground_truth_map(gt_file: Path, s1_filter_ids: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    """
    Loads ground truth mapping: source1_entity_id -> set of matched entity_ids.
    Optionally filters only relevant S1 IDs to save memory.
    """
    gt_map = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            if s1_filter_ids is not None and s1_id not in s1_filter_ids:
                continue
            matched_raw = row.get("matched_entity_ids", "")
            if matched_raw:
                matches = {m.strip() for m in matched_raw.split(",") if m.strip()}
                gt_map[s1_id] = matches
    return gt_map


def evaluate_blocking_strategy(
    strategy_fn,
    s1_records: List[dict],
    s2_index,
    s3_index,
    gt_map: Dict[str, Set[str]],
    indexed_target_ids: Optional[Set[str]] = None,
    top_k: int = 50
) -> dict:
    """
    Evaluates a candidate generation strategy against ground truth.
    Strictly asserts that recall is measured only over ground truth pairs whose
    target entity was present in the indexed retrieval universe.
    """
    t0 = time.perf_counter()
    mem_start = get_process_memory_mb()
    
    total_gt_pairs = 0
    recovered_gt_pairs = 0
    
    total_s2_gt_pairs = 0
    recovered_s2_gt_pairs = 0
    
    total_s3_gt_pairs = 0
    recovered_s3_gt_pairs = 0
    
    s1_with_gt_count = 0
    s1_with_at_least_one_recovered = 0
    
    candidate_counts = []
    
    for s1 in s1_records:
        s1_id = s1["entity_id"]
        raw_matches = gt_map.get(s1_id, set())
        
        # Filter to only targets that were actually indexed in the retrieval universe
        if indexed_target_ids is not None:
            true_matches = {m for m in raw_matches if m in indexed_target_ids}
        else:
            true_matches = raw_matches
        
        # Generate candidates
        cands = strategy_fn(s1, s2_index, s3_index, top_k=top_k)
        cand_set = set(cands)
        candidate_counts.append(len(cand_set))
        
        if true_matches:
            s1_with_gt_count += 1
            matched_in_cands = true_matches & cand_set
            recovered_gt_pairs += len(matched_in_cands)
            total_gt_pairs += len(true_matches)
            
            if len(matched_in_cands) > 0:
                s1_with_at_least_one_recovered += 1
                
            # Breakdown S2 vs S3
            for m in true_matches:
                if m.startswith("S2-"):
                    total_s2_gt_pairs += 1
                    if m in cand_set:
                        recovered_s2_gt_pairs += 1
                elif m.startswith("S3-"):
                    total_s3_gt_pairs += 1
                    if m in cand_set:
                        recovered_s3_gt_pairs += 1
                        
    t1 = time.perf_counter()
    mem_end = get_process_memory_mb()
    query_time_sec = round(t1 - t0, 4)
    qps = round(len(s1_records) / max(0.0001, query_time_sec), 2)
    
    # Compute recall metrics
    overall_recall = round(recovered_gt_pairs / max(1, total_gt_pairs), 6)
    s2_recall = round(recovered_s2_gt_pairs / max(1, total_s2_gt_pairs), 6)
    s3_recall = round(recovered_s3_gt_pairs / max(1, total_s3_gt_pairs), 6)
    s1_coverage = round(s1_with_at_least_one_recovered / max(1, s1_with_gt_count), 6)
    
    # Candidate volume metrics
    cand_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
    total_candidates_generated = int(np.sum(cand_arr))
    
    metrics = {
        "true_pair_recall": overall_recall,
        "s2_recall": s2_recall,
        "s3_recall": s3_recall,
        "s1_entity_coverage": s1_coverage,
        "total_gt_pairs": total_gt_pairs,
        "recovered_gt_pairs": recovered_gt_pairs,
        "candidate_volume": {
            "total_candidate_pairs": total_candidates_generated,
            "mean_per_s1": round(float(np.mean(cand_arr)), 2),
            "median_per_s1": float(np.median(cand_arr)),
            "p90_per_s1": float(np.percentile(cand_arr, 90)),
            "p95_per_s1": float(np.percentile(cand_arr, 95)),
            "p99_per_s1": float(np.percentile(cand_arr, 99)),
            "max_per_s1": int(np.max(cand_arr)),
            "min_per_s1": int(np.min(cand_arr))
        },
        "performance": {
            "query_time_sec": query_time_sec,
            "queries_per_sec": qps,
            "s1_queries_evaluated": len(s1_records),
            "memory_rss_start_mb": mem_start,
            "memory_rss_end_mb": mem_end,
            "memory_delta_mb": round(mem_end - mem_start, 2)
        }
    }
    return metrics
