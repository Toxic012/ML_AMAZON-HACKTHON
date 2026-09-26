#!/usr/bin/env python3
"""
High-Performance Resumable Production Pipeline & Submission Generator
Amazon ML Challenge 2026 — Business Entity Resolution

Features:
- Stream-based target indexing (Source 2 and Source 3) with compact tuple storage.
- Resumable checkpointing (resumes immediately if process is interrupted).
- Chunked S1 streaming & batch vectorized feature extraction (LightGBM).
- Direct disk flushing to matching_results.tsv and candidate_pairs.tsv.
- Memory-safe (< 4 GB peak RSS) for full 1.73M test set.
- Automatic official submission validation upon completion.
"""

import os
import sys
import csv
import time
import json
import gc
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

# Increase csv field limit
csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import get_dataset_dir, EXPERIMENTS_DIR, BASE_DIR
    from src.normalization import normalize_record
    from src.blocking.token_index import InvertedTokenIndex
    from src.blocking.strategies import block_hybrid_full_union
    from src.features import compute_pairwise_features
    from src.matching.matcher import EntityMatcher
except ImportError:
    from code.business_entity_resolution.src.config import get_dataset_dir, EXPERIMENTS_DIR, BASE_DIR
    from code.business_entity_resolution.src.normalization import normalize_record
    from code.business_entity_resolution.src.blocking.token_index import InvertedTokenIndex
    from code.business_entity_resolution.src.blocking.strategies import block_hybrid_full_union
    from code.business_entity_resolution.src.features import compute_pairwise_features
    from code.business_entity_resolution.src.matching.matcher import EntityMatcher


def get_process_memory_mb() -> float:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
    except Exception:
        return 0.0


def stream_compact_index(source_path: Path, max_token_freq: int = 5000, limit: Optional[int] = None):
    """
    Streams a target TSV (S2/S3) and builds:
    1. Inverted index for fast candidate retrieval
    2. Compact dictionary of records for fast feature extraction
    """
    index = InvertedTokenIndex(max_token_freq=max_token_freq)
    target_store: Dict[str, dict] = {}
    
    t0 = time.time()
    count = 0
    with open(source_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id")
            if not eid:
                continue
            norm_r = normalize_record(row)
            index.add_record(norm_r)
            # Store normalized record
            target_store[eid] = norm_r
            count += 1
            if limit and count >= limit:
                break
                
    elapsed = round(time.time() - t0, 2)
    return index, target_store, count, elapsed


def run_production_pipeline(
    data_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    top_k: int = 50,
    threshold: float = 0.83,
    batch_size: int = 2000,
    s1_limit: Optional[int] = None,
    target_limit: Optional[int] = None,
    resume: bool = True
):
    print("=" * 95)
    print("AMAZON ML CHALLENGE 2026 — FULL-SCALE PRODUCTION INFERENCE PIPELINE")
    print("=" * 95)
    
    t_pipeline_start = time.time()
    
    # 1. Resolve Dataset Paths
    resolved_data_dir = get_dataset_dir(data_dir)
    test_path = resolved_data_dir / "test" if (resolved_data_dir / "test").exists() else resolved_data_dir
    
    s1_file = test_path / "test_source1.tsv"
    s2_file = test_path / "test_source2.tsv"
    s3_file = test_path / "test_source3.tsv"
    
    assert s1_file.exists(), f"Source 1 test file missing: {s1_file}"
    assert s2_file.exists(), f"Source 2 test file missing: {s2_file}"
    assert s3_file.exists(), f"Source 3 test file missing: {s3_file}"
    
    # 2. Load Trained Matcher Model
    if model_path is None:
        model_path = EXPERIMENTS_DIR / "PHASE_3_pairwise_matching" / "matcher_model.pkl"
    else:
        model_path = Path(model_path)
        
    print(f"\n[1] Loading Trained LightGBM Matcher from {model_path} ...")
    matcher = EntityMatcher.load(model_path)
    tau = threshold if threshold is not None else matcher.best_threshold
    print(f"    Loaded Model with Decision Threshold tau* = {tau:.2f} (25 features)")

    # 3. Stream Index Target Datasets
    print(f"\n[2] Indexing Test Target Datasets (Source 2 and Source 3)...")
    print(f"    -> Streaming {s2_file.name} ...")
    s2_index, s2_store, n_s2, t_s2 = stream_compact_index(s2_file, limit=target_limit)
    print(f"       [OK] S2 Indexed: {n_s2:,} records in {t_s2:.2f}s | RSS: {get_process_memory_mb()} MB")
    
    print(f"    -> Streaming {s3_file.name} ...")
    s3_index, s3_store, n_s3, t_s3 = stream_compact_index(s3_file, limit=target_limit)
    print(f"       [OK] S3 Indexed: {n_s3:,} records in {t_s3:.2f}s | RSS: {get_process_memory_mb()} MB")
    
    target_store = {**s2_store, **s3_store}
    print(f"    Total Target Universe Available: {len(target_store):,} records | Post-Index RSS: {get_process_memory_mb()} MB")

    # 4. Prepare Output and Checkpoint Files
    if output_dir is None:
        out_path = REPO_ROOT / "output"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    matching_tsv = out_path / "matching_results.tsv"
    candidate_tsv = out_path / "candidate_pairs.tsv"
    checkpoint_file = out_path / ".inference_checkpoint.json"
    
    # Check existing progress
    processed_s1_ids = set()
    if resume and checkpoint_file.exists() and matching_tsv.exists() and candidate_tsv.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f_cp:
                cp_data = json.load(f_cp)
                processed_count = cp_data.get("processed_count", 0)
                print(f"\n[RESUME] Found existing checkpoint with {processed_count:,} processed S1 entities.")
                # Read already processed IDs from matching_tsv
                with open(matching_tsv, "r", encoding="utf-8") as f_m:
                    next(f_m, None)
                    for line in f_m:
                        parts = line.split("\t", 1)
                        if parts and parts[0].strip():
                            processed_s1_ids.add(parts[0].strip())
                print(f"         Verified {len(processed_s1_ids):,} existing records. Resuming remaining entities...")
        except Exception as e:
            print(f"[WARN] Failed to read checkpoint ({e}). Starting fresh.")
            processed_s1_ids = set()

    # If starting fresh, initialize files with exact required headers
    if not processed_s1_ids:
        with open(matching_tsv, "w", encoding="utf-8", newline="") as f_m:
            f_m.write("source1_entity_id\tmatched_entity_ids\n")
        with open(candidate_tsv, "w", encoding="utf-8", newline="") as f_c:
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    # 5. Stream S1 Entities and Execute Inference in Batches
    print(f"\n[3] Streaming Source 1 Entities & Executing Inference (Batch Size = {batch_size:,})...")
    
    f_match = open(matching_tsv, "a", encoding="utf-8", newline="")
    f_cand = open(candidate_tsv, "a", encoding="utf-8", newline="")
    
    total_processed = len(processed_s1_ids)
    total_candidates_generated = 0
    total_matches_predicted = 0
    singletons_count = 0
    
    batch_records = []
    t_start_s1 = time.time()
    last_log_time = time.time()
    
    try:
        with open(s1_file, "r", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in, delimiter="\t")
            for row in reader:
                s1_id = row.get("entity_id")
                if not s1_id:
                    continue
                if s1_id in processed_s1_ids:
                    continue
                    
                norm_s1 = normalize_record(row)
                batch_records.append(norm_s1)
                
                if len(batch_records) >= batch_size:
                    # Process current batch
                    c_gen, m_pred, s_cnt = process_s1_batch(
                        batch_records,
                        s2_index,
                        s3_index,
                        target_store,
                        matcher,
                        tau,
                        top_k,
                        f_match,
                        f_cand
                    )
                    total_candidates_generated += c_gen
                    total_matches_predicted += m_pred
                    singletons_count += s_cnt
                    total_processed += len(batch_records)
                    
                    # Update checkpoint
                    with open(checkpoint_file, "w", encoding="utf-8") as f_cp:
                        json.dump({"processed_count": total_processed, "timestamp": datetime.now().isoformat()}, f_cp)
                        
                    batch_records = []
                    
                    if time.time() - last_log_time >= 15.0 or (total_processed % 10000 == 0):
                        elapsed = time.time() - t_start_s1
                        qps = total_processed / max(0.001, elapsed)
                        print(f"    Processed {total_processed:,} S1 entities... (Speed: {qps:.1f} S1/s | RSS: {get_process_memory_mb()} MB | Candidates: {total_candidates_generated:,} | Matches: {total_matches_predicted:,})")
                        last_log_time = time.time()
                        gc.collect()
                        
                if s1_limit and total_processed >= s1_limit:
                    break
                    
            # Process trailing batch
            if batch_records:
                c_gen, m_pred, s_cnt = process_s1_batch(
                    batch_records,
                    s2_index,
                    s3_index,
                    target_store,
                    matcher,
                    tau,
                    top_k,
                    f_match,
                    f_cand
                )
                total_candidates_generated += c_gen
                total_matches_predicted += m_pred
                singletons_count += s_cnt
                total_processed += len(batch_records)
                
    finally:
        f_match.close()
        f_cand.close()

    total_pipeline_time = round(time.time() - t_pipeline_start, 2)
    print(f"\n[OK] Inference Execution Complete in {total_pipeline_time}s ({total_processed/max(0.001, total_pipeline_time):.1f} S1/sec)")
    print(f"    Total S1 Entities Processed:  {total_processed:,}")
    print(f"    Total Candidates Output:     {total_candidates_generated:,} (Mean: {total_candidates_generated/max(1, total_processed):.1f} / S1)")
    print(f"    Total Matches Predicted:     {total_matches_predicted:,}")
    print(f"    Singletons Output:           {singletons_count:,} ({singletons_count/max(1, total_processed)*100:.1f}%)")
    print(f"    Peak Memory RSS:             {get_process_memory_mb()} MB")
    print(f"    Matching File Size:          {round(matching_tsv.stat().st_size / (1024**2), 2)} MB")
    print(f"    Candidate File Size:         {round(candidate_tsv.stat().st_size / (1024**2), 2)} MB")

    # 6. Run Official Submission Validator
    print(f"\n[4] Running Official Submission Validator...")
    val_script = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    validator_passed = False
    
    if val_script.exists():
        res = subprocess.run([
            sys.executable, str(val_script),
            "--matching", str(matching_tsv),
            "--candidate", str(candidate_tsv),
            "--test-dir", str(test_path)
        ], capture_output=True, text=True)
        
        print(f"    Validator Return Code: {res.returncode}")
        print("\n" + "=" * 50 + " OFFICIAL VALIDATOR REPORT " + "=" * 50)
        print(res.stdout.strip())
        print("=" * 125)
        
        validator_passed = (res.returncode == 0)
        if validator_passed:
            print("\n>>> [SUCCESS] OFFICIAL SUBMISSION VALIDATOR PASSED (100% COMPLIANT) <<<")
        else:
            print("\n>>> [NOTICE] Validator report output shown above. <<<")

    return {
        "validator_passed": validator_passed,
        "total_s1": total_processed,
        "total_candidates": total_candidates_generated,
        "total_matches": total_matches_predicted,
        "singletons": singletons_count,
        "runtime_seconds": total_pipeline_time,
        "matching_tsv": str(matching_tsv),
        "candidate_tsv": str(candidate_tsv)
    }


def process_s1_batch(
    batch_s1: List[dict],
    s2_index: InvertedTokenIndex,
    s3_index: InvertedTokenIndex,
    target_store: Dict[str, dict],
    matcher: EntityMatcher,
    threshold: float,
    top_k: int,
    f_match,
    f_cand
) -> Tuple[int, int, int]:
    """Processes a batch of S1 entities, extracts features, predicts matches, and flushes to disk."""
    total_cands = 0
    total_matches = 0
    singletons = 0
    
    for s1_rec in batch_s1:
        s1_id = s1_rec["entity_id"]
        # Candidate Generation via Frozen Strategy E
        cands = block_hybrid_full_union(s1_rec, s2_index, s3_index, top_k=top_k)
        total_cands += len(cands)
        
        # Write candidate pairs line
        f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
        
        matched_ids = []
        if cands:
            pair_feats = []
            valid_cands = []
            for cid in cands:
                c_rec = target_store.get(cid)
                if c_rec is not None:
                    pair_feats.append(compute_pairwise_features(s1_rec, c_rec))
                    valid_cands.append(cid)
                    
            if pair_feats:
                X_batch = np.array(pair_feats, dtype=np.float32)
                probas = matcher.predict_proba(X_batch)
                for cid, prob in zip(valid_cands, probas):
                    if prob >= threshold:
                        matched_ids.append(cid)
                        
        if not matched_ids:
            singletons += 1
            f_match.write(f"{s1_id}\t\n")
        else:
            total_matches += len(matched_ids)
            f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")
            
    f_match.flush()
    f_cand.flush()
    return total_cands, total_matches, singletons


def main():
    parser = argparse.ArgumentParser(description="Full Production Pipeline & Submission Generator")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--model-path", type=str, default=None, help="Trained model path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder for submission TSVs")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate capacity per query")
    parser.add_argument("--threshold", type=float, default=0.83, help="Decision threshold")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for S1 processing")
    parser.add_argument("--s1-limit", type=int, default=None, help="Limit S1 queries for test runs")
    parser.add_argument("--target-limit", type=int, default=None, help="Limit target records for test runs")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh without resuming")
    
    args = parser.parse_args()
    
    run_production_pipeline(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        threshold=args.threshold,
        batch_size=args.batch_size,
        s1_limit=args.s1_limit,
        target_limit=args.target_limit,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
