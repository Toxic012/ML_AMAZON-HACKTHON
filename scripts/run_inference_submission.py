#!/usr/bin/env python3
"""
Production Inference & Submission Generator
Amazon ML Challenge 2026 — Business Entity Resolution

Workflow:
1. Loads trained EntityMatcher model from experiments/PHASE_3_pairwise_matching/matcher_model.pkl
2. Indexes Target Sources (Source 2 and Source 3)
3. Streams Source 1 queries in chunks
4. Generates candidate pairs using frozen Strategy E (Hybrid Full Union)
5. Extracts 25-dim pairwise features
6. Predicts match probabilities and applies optimal threshold tau*
7. Writes production matching_results.tsv and candidate_pairs.tsv
8. Runs official submission validator and verifies PASS
"""

import os
import sys
import csv
import time
import argparse
import subprocess
from pathlib import Path
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
    from src.config import get_dataset_dir, EXPERIMENTS_DIR
    from src.normalization import normalize_record
    from src.blocking.token_index import InvertedTokenIndex
    from src.blocking.strategies import block_hybrid_full_union
    from src.features import compute_pairwise_features
    from src.matching.matcher import EntityMatcher
except ImportError:
    from code.business_entity_resolution.src.config import get_dataset_dir, EXPERIMENTS_DIR
    from code.business_entity_resolution.src.normalization import normalize_record
    from code.business_entity_resolution.src.blocking.token_index import InvertedTokenIndex
    from code.business_entity_resolution.src.blocking.strategies import block_hybrid_full_union
    from code.business_entity_resolution.src.features import compute_pairwise_features
    from code.business_entity_resolution.src.matching.matcher import EntityMatcher


def stream_index_source(source_path: Path, max_token_freq: int = 5000, limit: Optional[int] = None):
    index = InvertedTokenIndex(max_token_freq=max_token_freq)
    target_store = {}
    
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
            target_store[eid] = norm_r
            count += 1
            if limit and count >= limit:
                break
                
    elapsed = round(time.time() - t0, 2)
    return index, target_store, count, elapsed


def run_inference(
    test_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    top_k: int = 50,
    threshold: Optional[float] = None,
    s1_limit: Optional[int] = None,
    target_limit: Optional[int] = None
):
    print("=" * 85)
    print("AMAZON ML CHALLENGE 2026 — PRODUCTION INFERENCE & SUBMISSION PIPELINE")
    print("=" * 85)
    
    resolved_data_dir = get_dataset_dir(test_dir)
    test_path = resolved_data_dir / "test" if (resolved_data_dir / "test").exists() else resolved_data_dir
    
    s1_file = test_path / "test_source1.tsv"
    s2_file = test_path / "test_source2.tsv"
    s3_file = test_path / "test_source3.tsv"
    
    assert s1_file.exists(), f"Source1 test file missing: {s1_file}"
    assert s2_file.exists(), f"Source2 test file missing: {s2_file}"
    assert s3_file.exists(), f"Source3 test file missing: {s3_file}"
    
    # 1. Load Trained Matcher Model
    if model_path is None:
        model_path = EXPERIMENTS_DIR / "PHASE_3_pairwise_matching" / "matcher_model.pkl"
    else:
        model_path = Path(model_path)
        
    print(f"\n[1] Loading Trained Matcher from {model_path} ...")
    matcher = EntityMatcher.load(model_path)
    tau = threshold if threshold is not None else matcher.best_threshold
    print(f"    Loaded Model with Decision Threshold tau* = {tau:.2f}")

    # 2. Index Target Test Datasets S2 & S3
    print(f"\n[2] Indexing Test Target Datasets (Source 2 and Source 3)...")
    s2_index, s2_store, n_s2, t_s2 = stream_index_source(s2_file, limit=target_limit)
    print(f"    [OK] Source 2: {n_s2:,} records indexed in {t_s2:.2f}s")
    
    s3_index, s3_store, n_s3, t_s3 = stream_index_source(s3_file, limit=target_limit)
    print(f"    [OK] Source 3: {n_s3:,} records indexed in {t_s3:.2f}s")
    
    target_store = {**s2_store, **s3_store}

    # 3. Setup Output Files
    if output_dir is None:
        out_path = REPO_ROOT / "output"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    matching_tsv = out_path / "matching_results.tsv"
    candidate_tsv = out_path / "candidate_pairs.tsv"
    
    print(f"\n[3] Streaming Source 1 Entities & Generating Predictions...")
    t_start = time.time()
    
    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    singleton_count = 0
    
    with open(s1_file, "r", encoding="utf-8") as f_in, \
         open(matching_tsv, "w", encoding="utf-8", newline="") as f_match, \
         open(candidate_tsv, "w", encoding="utf-8", newline="") as f_cand:
         
        # Write exact required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        reader = csv.DictReader(f_in, delimiter="\t")
        
        batch_s1 = []
        BATCH_SIZE = 1000
        
        for row in reader:
            total_s1 += 1
            norm_s1 = normalize_record(row)
            batch_s1.append(norm_s1)
            
            if len(batch_s1) >= BATCH_SIZE:
                # Process batch
                for s1_rec in batch_s1:
                    s1_id = s1_rec["entity_id"]
                    cands = block_hybrid_full_union(s1_rec, s2_index, s3_index, top_k=top_k)
                    total_candidates += len(cands)
                    f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
                    
                    matched_ids = []
                    if cands:
                        pair_feats = []
                        valid_cands = []
                        for cid in cands:
                            c_rec = target_store.get(cid)
                            if c_rec:
                                pair_feats.append(compute_pairwise_features(s1_rec, c_rec))
                                valid_cands.append(cid)
                        if pair_feats:
                            X_batch = np.array(pair_feats, dtype=np.float32)
                            probas = matcher.predict_proba(X_batch)
                            for cid, prob in zip(valid_cands, probas):
                                if prob >= tau:
                                    matched_ids.append(cid)
                                    
                    if not matched_ids:
                        singleton_count += 1
                        f_match.write(f"{s1_id}\t\n")
                    else:
                        total_matches += len(matched_ids)
                        f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")
                        
                batch_s1 = []
                if total_s1 % 5000 == 0:
                    print(f"    Processed {total_s1:,} S1 entities... (Total candidates: {total_candidates:,}, matches: {total_matches:,})")
                    
            if s1_limit and total_s1 >= s1_limit:
                break
                
        # Process remaining
        for s1_rec in batch_s1:
            s1_id = s1_rec["entity_id"]
            cands = block_hybrid_full_union(s1_rec, s2_index, s3_index, top_k=top_k)
            total_candidates += len(cands)
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
            
            matched_ids = []
            if cands:
                pair_feats = []
                valid_cands = []
                for cid in cands:
                    c_rec = target_store.get(cid)
                    if c_rec:
                        pair_feats.append(compute_pairwise_features(s1_rec, c_rec))
                        valid_cands.append(cid)
                if pair_feats:
                    X_batch = np.array(pair_feats, dtype=np.float32)
                    probas = matcher.predict_proba(X_batch)
                    for cid, prob in zip(valid_cands, probas):
                        if prob >= tau:
                            matched_ids.append(cid)
                            
            if not matched_ids:
                singleton_count += 1
                f_match.write(f"{s1_id}\t\n")
            else:
                total_matches += len(matched_ids)
                f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")

    total_time = time.time() - t_start
    print(f"\n[OK] Inference Completed in {total_time:.2f}s ({total_s1/max(0.001, total_time):.1f} S1/sec)")
    print(f"    Total S1 Entities Processed:  {total_s1:,}")
    print(f"    Total Candidate Pairs:       {total_candidates:,} (Mean: {total_candidates/max(1, total_s1):.1f}/S1)")
    print(f"    Total Predicted Matches:     {total_matches:,}")
    print(f"    Predicted Singletons:        {singleton_count:,} ({singleton_count/max(1, total_s1)*100:.1f}%)")
    print(f"    Written: {matching_tsv}")
    print(f"    Written: {candidate_tsv}")

    # 4. Run Official Submission Validator
    print(f"\n[4] Running Official Submission Validator...")
    val_script = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    if val_script.exists():
        res = subprocess.run([
            sys.executable, str(val_script),
            "--matching", str(matching_tsv),
            "--candidate", str(candidate_tsv),
            "--test-dir", str(test_path)
        ], capture_output=True, text=True)
        
        print(f"    Validator Return Code: {res.returncode}")
        print("\n" + "=" * 50 + " VALIDATOR REPORT " + "=" * 50)
        print(res.stdout.strip())
        print("=" * 118)
        
        if res.returncode == 0:
            print("\n>>> [SUCCESS] SUBMISSION VALIDATOR PASSED! Output is 100% compliant and ready for leaderboard.")
        else:
            print("\n>>> [FAIL] Submission Validator reported issues.")
            
    return {
        "total_s1": total_s1,
        "total_candidates": total_candidates,
        "total_matches": total_matches,
        "singleton_count": singleton_count,
        "matching_tsv": str(matching_tsv),
        "candidate_tsv": str(candidate_tsv)
    }


def main():
    parser = argparse.ArgumentParser(description="Production Inference & Submission Generator")
    parser.add_argument("--test-dir", type=str, default=None, help="Test dataset directory")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained matcher_model.pkl")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for submission TSVs")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate budget per query")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override")
    parser.add_argument("--s1-limit", type=int, default=None, help="Limit number of S1 entities (for validation/testing)")
    parser.add_argument("--target-limit", type=int, default=None, help="Limit target records")
    
    args = parser.parse_args()
    
    run_inference(
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        threshold=args.threshold,
        s1_limit=args.s1_limit,
        target_limit=args.target_limit
    )


if __name__ == "__main__":
    main()
