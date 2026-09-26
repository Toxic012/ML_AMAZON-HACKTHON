#!/usr/bin/env python3
"""
High-Performance Sequential Production Pipeline & Submission Generator
Amazon ML Challenge 2026 — Business Entity Resolution

Architecture:
1. Phase 1 (Source 2):
   - Index Target Source 2 into compact memory representation.
   - Stream S1 queries through Strategy E (Source 2 quota: K/2=25) & LightGBM scoring.
   - Stream matches & candidates to intermediate/s2_matches.tsv and intermediate/s2_candidates.tsv.
   - Explicitly delete Source 2 index & target store, trigger GC, log freed RAM.
2. Phase 2 (Source 3):
   - Index Target Source 3 into compact memory representation.
   - Stream S1 queries through Strategy E (Source 3 quota: K/2=25) & LightGBM scoring.
   - Stream matches & candidates to intermediate/s3_matches.tsv and intermediate/s3_candidates.tsv.
   - Explicitly delete Source 3 index & target store, trigger GC, log freed RAM.
3. Phase 3 (Stream Merge):
   - Stream-merge S2 and S3 outputs line-by-line into matching_results.tsv and candidate_pairs.tsv.
   - Peak RSS during merge < 50 MB.
4. Phase 4 (Validation):
   - Automatic submission validation via official student_resource validator.
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
    from src.config import (
        get_dataset_dir,
        resolve_dataset_paths,
        print_dataset_diagnostics,
        stage_test_files_locally,
        EXPERIMENTS_DIR,
        BASE_DIR
    )
    from src.normalization import normalize_record, normalize_text
    from src.blocking.token_index import InvertedTokenIndex
    from src.blocking.strategies import block_hybrid_single_source
    from src.features import compute_pairwise_features
    from src.matching.matcher import EntityMatcher
except ImportError:
    from code.business_entity_resolution.src.config import (
        get_dataset_dir,
        resolve_dataset_paths,
        print_dataset_diagnostics,
        stage_test_files_locally,
        EXPERIMENTS_DIR,
        BASE_DIR
    )
    from code.business_entity_resolution.src.normalization import normalize_record, normalize_text
    from code.business_entity_resolution.src.blocking.token_index import InvertedTokenIndex
    from code.business_entity_resolution.src.blocking.strategies import block_hybrid_single_source
    from code.business_entity_resolution.src.features import compute_pairwise_features
    from code.business_entity_resolution.src.matching.matcher import EntityMatcher


def get_process_memory_mb() -> float:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
    except Exception:
        return 0.0


def count_lines_fast(file_path: Path) -> int:
    """Memory-safe line counter using chunked buffered reads."""
    count = 0
    with open(file_path, "rb") as f:
        buffer_size = 1024 * 1024
        while chunk := f.read(buffer_size):
            count += chunk.count(b"\n")
    return count


def stream_compact_index(source_path: Path, max_token_freq: int = 5000, limit: Optional[int] = None, log_every: int = 250000):
    """
    Streams a target TSV (S2 or S3) and builds:
    1. Inverted index for fast candidate retrieval.
    2. Compact tuple dictionary (norm_name, norm_addr, country) for fast feature extraction.
    With periodic diagnostic progress logging.
    """
    index = InvertedTokenIndex(max_token_freq=max_token_freq)
    target_store: Dict[str, Tuple[str, str, str]] = {}
    
    t0 = time.time()
    t_last = t0
    count = 0
    source_name = source_path.name
    with open(source_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id")
            if not eid:
                continue
            name = row.get("business_name", "")
            addr = row.get("business_address", "")
            country = row.get("country", "").strip().upper()
            norm_name = normalize_text(name)
            norm_addr = normalize_text(addr)
            
            index.add_compact_record(eid, norm_name, norm_addr, country)
            target_store[eid] = (norm_name, norm_addr, country)
            count += 1
            
            if count % log_every == 0:
                now = time.time()
                elapsed = now - t0
                step_elapsed = now - t_last
                step_rate = log_every / max(0.001, step_elapsed)
                avg_rate = count / max(0.001, elapsed)
                rss = get_process_memory_mb()
                print(f"    [{source_name} INDEX] rows={count:>9,} | elapsed={elapsed:6.1f}s | step_rate={step_rate:>7,.0f} rows/s | avg_rate={avg_rate:>7,.0f} rows/s | RSS={rss:6.1f} MB")
                t_last = now
                
            if limit and count >= limit:
                break
                
    elapsed = round(time.time() - t0, 2)
    return index, target_store, count, elapsed


def process_single_source_phase(
    source_label: str,
    source_file: Path,
    s1_file: Path,
    matcher: EntityMatcher,
    threshold: float,
    top_k_source: int,
    intermediate_dir: Path,
    batch_size: int = 2000,
    s1_limit: Optional[int] = None,
    target_limit: Optional[int] = None,
    resume: bool = True
) -> Tuple[Path, Path, float, float]:
    """
    Executes a complete isolated inference pass for one target source (S2 or S3):
    1. Indexes target source in compact memory representation.
    2. Streams all S1 entities, extracts top_k_source candidates, scores with LightGBM.
    3. Streams matches and candidates incrementally to intermediate TSVs.
    4. Deletes target index & store, triggers garbage collection, measures memory drop.
    Returns (matches_tsv_path, candidates_tsv_path, peak_rss_mb, post_cleanup_rss_mb).
    """
    matches_tsv = intermediate_dir / f"{source_label.lower()}_matches.tsv"
    candidates_tsv = intermediate_dir / f"{source_label.lower()}_candidates.tsv"
    checkpoint_file = intermediate_dir / f".{source_label.lower()}_checkpoint.json"
    
    # Check if already completed under resume mode
    if resume and matches_tsv.exists() and candidates_tsv.exists() and checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f_cp:
                cp = json.load(f_cp)
                if cp.get("status") == "COMPLETED":
                    n_rows = cp.get("processed_count", 0)
                    print(f"[{source_label.upper()} PHASE] Found completed intermediate results ({n_rows:,} S1 rows). Skipping re-computation.")
                    return matches_tsv, candidates_tsv, get_process_memory_mb(), get_process_memory_mb()
        except Exception:
            pass

    print("=" * 90)
    print(f"[{source_label.upper()} PHASE] Indexing & Inference for {source_file.name}")
    print("=" * 90)
    
    rss_phase_start = get_process_memory_mb()
    print(f"  [1] Building {source_label.upper()} Inverted Index from {source_file.name} (Start RSS: {rss_phase_start} MB)...")
    target_index, target_store, n_target, t_index = stream_compact_index(source_file, limit=target_limit)
    rss_post_index = get_process_memory_mb()
    print(f"      [OK] {source_label.upper()} Indexed: {n_target:,} records in {t_index:.2f}s | RSS: {rss_post_index} MB (Delta: +{rss_post_index - rss_phase_start:.1f} MB)")
    
    # Prepare intermediate TSVs
    with open(matches_tsv, "w", encoding="utf-8", newline="") as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
    with open(candidates_tsv, "w", encoding="utf-8", newline="") as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
    f_match = open(matches_tsv, "a", encoding="utf-8", newline="")
    f_cand = open(candidates_tsv, "a", encoding="utf-8", newline="")
    
    total_processed = 0
    total_candidates = 0
    total_matches = 0
    singletons = 0
    batch_records = []
    
    t_start = time.time()
    last_log_time = time.time()
    
    print(f"\n  [2] Streaming S1 Queries against {source_label.upper()} (Top-K per source = {top_k_source}, Threshold = {threshold:.2f})...")
    
    try:
        with open(s1_file, "r", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in, delimiter="\t")
            for row in reader:
                s1_id = row.get("entity_id")
                if not s1_id:
                    continue
                    
                norm_s1 = normalize_record(row)
                batch_records.append(norm_s1)
                
                if len(batch_records) >= batch_size:
                    c_cnt, m_cnt, s_cnt = _process_source_batch(
                        batch_records,
                        target_index,
                        target_store,
                        matcher,
                        threshold,
                        top_k_source,
                        f_match,
                        f_cand
                    )
                    total_candidates += c_cnt
                    total_matches += m_cnt
                    singletons += s_cnt
                    total_processed += len(batch_records)
                    batch_records = []
                    
                    if time.time() - last_log_time >= 15.0 or (total_processed % 25000 == 0):
                        elapsed = time.time() - t_start
                        qps = total_processed / max(0.001, elapsed)
                        print(f"      Processed {total_processed:>9,} S1 entities... (Speed: {qps:6.1f} S1/s | RSS: {get_process_memory_mb():6.1f} MB | {source_label.upper()} Cands: {total_candidates:,} | {source_label.upper()} Matches: {total_matches:,})")
                        last_log_time = time.time()
                        
                if s1_limit and total_processed >= s1_limit:
                    break
                    
            if batch_records:
                c_cnt, m_cnt, s_cnt = _process_source_batch(
                    batch_records,
                    target_index,
                    target_store,
                    matcher,
                    threshold,
                    top_k_source,
                    f_match,
                    f_cand
                )
                total_candidates += c_cnt
                total_matches += m_cnt
                singletons += s_cnt
                total_processed += len(batch_records)
                
    finally:
        f_match.close()
        f_cand.close()
        
    peak_rss = get_process_memory_mb()
    total_elapsed = time.time() - t_start
    print(f"\n  [OK] {source_label.upper()} Inference Finished: {total_processed:,} S1 entities in {total_elapsed:.2f}s ({total_processed/max(0.001, total_elapsed):.1f} S1/s)")
    print(f"       Total {source_label.upper()} Candidates: {total_candidates:,} | Total {source_label.upper()} Matches: {total_matches:,} | Peak RSS: {peak_rss:.1f} MB")
    
    # Save checkpoint
    with open(checkpoint_file, "w", encoding="utf-8") as f_cp:
        json.dump({
            "status": "COMPLETED",
            "source": source_label,
            "processed_count": total_processed,
            "candidates_count": total_candidates,
            "matches_count": total_matches,
            "timestamp": datetime.now().isoformat()
        }, f_cp)

    # 3. Clean up Index and Target Store from RAM
    print(f"\n  [3] Freeing {source_label.upper()} Index and Target Store from RAM (Current RSS: {peak_rss:.1f} MB)...")
    del target_index
    del target_store
    gc.collect()
    
    post_cleanup_rss = get_process_memory_mb()
    freed_mb = peak_rss - post_cleanup_rss
    print(f"      [AFTER {source_label.upper()} CLEANUP] RSS: {peak_rss:.1f} MB -> {post_cleanup_rss:.1f} MB (Successfully Freed: {freed_mb:.1f} MB)")
    print("-" * 90 + "\n")
    
    return matches_tsv, candidates_tsv, peak_rss, post_cleanup_rss


def _process_source_batch(
    batch_s1: List[dict],
    target_index: InvertedTokenIndex,
    target_store: Dict[str, Tuple[str, str, str]],
    matcher: EntityMatcher,
    threshold: float,
    top_k_source: int,
    f_match,
    f_cand
) -> Tuple[int, int, int]:
    """Evaluates a batch of S1 queries against one target index, writes intermediate TSV lines."""
    total_cands = 0
    total_matches = 0
    singletons = 0
    
    for s1_rec in batch_s1:
        s1_id = s1_rec["entity_id"]
        # Candidate Generation via Strategy E for single source
        cands = block_hybrid_single_source(s1_rec, target_index, top_k=top_k_source)
        total_cands += len(cands)
        
        # Write intermediate candidate line
        f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
        
        matched_ids = []
        if cands:
            pair_feats = []
            valid_cands = []
            for cid in cands:
                c_rec = target_store.get(cid)
                if c_rec is not None:
                    pair_feats.append(compute_pairwise_features(s1_rec, c_rec, cand_id=cid))
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


def merge_source_predictions(
    s2_match_tsv: Path,
    s2_cand_tsv: Path,
    s3_match_tsv: Path,
    s3_cand_tsv: Path,
    final_matching_tsv: Path,
    final_candidate_tsv: Path
) -> Tuple[int, int, int, int]:
    """
    Stream-merges intermediate S2 and S3 output TSVs into official competition submission format.
    Memory-safe (O(1) RAM) streaming line-by-line.
    """
    print("=" * 90)
    print("[PHASE 3/3] Stream-Merging S2 and S3 Predictions into Final Submission TSVs")
    print("=" * 90)
    t0 = time.time()
    
    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    singletons = 0
    
    # 1. Merge Matches
    print(f"  -> Merging Matches: {s2_match_tsv.name} + {s3_match_tsv.name} -> {final_matching_tsv.name} ...")
    with open(s2_match_tsv, "r", encoding="utf-8") as f2_m, \
         open(s3_match_tsv, "r", encoding="utf-8") as f3_m, \
         open(final_matching_tsv, "w", encoding="utf-8", newline="") as f_out_m:
         
        next(f2_m, None)
        next(f3_m, None)
        f_out_m.write("source1_entity_id\tmatched_entity_ids\n")
        
        for l2, l3 in zip(f2_m, f3_m):
            p2 = l2.rstrip("\r\n").split("\t", 1)
            p3 = l3.rstrip("\r\n").split("\t", 1)
            s1_id = p2[0]
            
            m2 = p2[1].split(",") if (len(p2) > 1 and p2[1]) else []
            m3 = p3[1].split(",") if (len(p3) > 1 and p3[1]) else []
            
            merged_matches = m2 + m3
            if merged_matches:
                total_matches += len(merged_matches)
                f_out_m.write(f"{s1_id}\t{','.join(merged_matches)}\n")
            else:
                singletons += 1
                f_out_m.write(f"{s1_id}\t\n")
            total_s1 += 1
            
    # 2. Merge Candidates
    print(f"  -> Merging Candidates: {s2_cand_tsv.name} + {s3_cand_tsv.name} -> {final_candidate_tsv.name} ...")
    with open(s2_cand_tsv, "r", encoding="utf-8") as f2_c, \
         open(s3_cand_tsv, "r", encoding="utf-8") as f3_c, \
         open(final_candidate_tsv, "w", encoding="utf-8", newline="") as f_out_c:
         
        next(f2_c, None)
        next(f3_c, None)
        f_out_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for l2, l3 in zip(f2_c, f3_c):
            p2 = l2.rstrip("\r\n").split("\t", 1)
            p3 = l3.rstrip("\r\n").split("\t", 1)
            s1_id = p2[0]
            
            c2 = p2[1].split(",") if (len(p2) > 1 and p2[1]) else []
            c3 = p3[1].split(",") if (len(p3) > 1 and p3[1]) else []
            
            merged_cands = c2 + c3
            total_candidates += len(merged_cands)
            f_out_c.write(f"{s1_id}\t{','.join(merged_cands)}\n")
            
    elapsed = time.time() - t0
    print(f"  [OK] Stream Merge Complete in {elapsed:.2f}s | S1: {total_s1:,} | Matches: {total_matches:,} | Candidates: {total_candidates:,} | Singletons: {singletons:,}\n")
    return total_s1, total_candidates, total_matches, singletons


def run_production_pipeline(
    data_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    top_k: int = 50,
    threshold: float = 0.83,
    batch_size: int = 2000,
    s1_limit: Optional[int] = None,
    target_limit: Optional[int] = None,
    stage_local: bool = True,
    resume: bool = True
):
    print("=" * 95)
    print("AMAZON ML CHALLENGE 2026 — FULL-SCALE SEQUENTIAL PRODUCTION INFERENCE PIPELINE")
    print("=" * 95)
    
    t_pipeline_start = time.time()
    
    # 1. Resolve Dataset Paths with Diagnostics
    resolved_paths = resolve_dataset_paths(data_dir)
    print_dataset_diagnostics(resolved_paths)
    
    # Optionally stage test files to local fast disk to avoid Drive FUSE latency
    if stage_local:
        resolved_paths = stage_test_files_locally(resolved_paths)
    
    s1_file = resolved_paths.get("test_source1")
    s2_file = resolved_paths.get("test_source2")
    s3_file = resolved_paths.get("test_source3")
    
    assert s1_file and s1_file.exists(), (
        f"[FATAL CONFIG ERROR] Test Source 1 file could not be resolved in {resolved_paths['data_dir']}!"
    )
    assert s2_file and s2_file.exists(), (
        f"[FATAL CONFIG ERROR] Test Source 2 file could not be resolved in {resolved_paths['data_dir']}!"
    )
    assert s3_file and s3_file.exists(), (
        f"[FATAL CONFIG ERROR] Test Source 3 file could not be resolved in {resolved_paths['data_dir']}!"
    )
    
    # 2. Load Trained Matcher Model
    if model_path is None:
        model_path = EXPERIMENTS_DIR / "PHASE_3_pairwise_matching" / "matcher_model.pkl"
    else:
        model_path = Path(model_path)
        
    print(f"\n[LOAD MODEL] Loading Trained LightGBM Matcher from {model_path} ...")
    matcher = EntityMatcher.load(model_path)
    tau = threshold if threshold is not None else matcher.best_threshold
    print(f"             Loaded Model with Decision Threshold tau* = {tau:.2f} (25 features)\n")

    # 3. Setup Output & Intermediate Directories
    if output_dir is None:
        out_path = REPO_ROOT / "output"
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    intermediate_dir = out_path / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    final_matching_tsv = out_path / "matching_results.tsv"
    final_candidate_tsv = out_path / "candidate_pairs.tsv"
    
    top_k_per_source = max(1, top_k // 2)

    # 4. Phase 1: Source 2 Processing
    s2_m_tsv, s2_c_tsv, s2_peak_rss, s2_post_rss = process_single_source_phase(
        source_label="S2",
        source_file=s2_file,
        s1_file=s1_file,
        matcher=matcher,
        threshold=tau,
        top_k_source=top_k_per_source,
        intermediate_dir=intermediate_dir,
        batch_size=batch_size,
        s1_limit=s1_limit,
        target_limit=target_limit,
        resume=resume
    )
    
    # 5. Phase 2: Source 3 Processing
    s3_m_tsv, s3_c_tsv, s3_peak_rss, s3_post_rss = process_single_source_phase(
        source_label="S3",
        source_file=s3_file,
        s1_file=s1_file,
        matcher=matcher,
        threshold=tau,
        top_k_source=top_k_per_source,
        intermediate_dir=intermediate_dir,
        batch_size=batch_size,
        s1_limit=s1_limit,
        target_limit=target_limit,
        resume=resume
    )
    
    # 6. Phase 3: Merge Predictions & Candidates
    total_s1, total_cands, total_matches, singletons = merge_source_predictions(
        s2_match_tsv=s2_m_tsv,
        s2_cand_tsv=s2_c_tsv,
        s3_match_tsv=s3_m_tsv,
        s3_cand_tsv=s3_c_tsv,
        final_matching_tsv=final_matching_tsv,
        final_candidate_tsv=final_candidate_tsv
    )
    
    total_pipeline_time = round(time.time() - t_pipeline_start, 2)
    
    print("=" * 90)
    print("PRODUCTION PIPELINE TELEMETRY SUMMARY")
    print("=" * 90)
    print(f"  Total Runtime:               {total_pipeline_time} seconds ({total_s1/max(0.001, total_pipeline_time):.1f} S1/sec)")
    print(f"  Total S1 Entities:           {total_s1:,}")
    print(f"  Total Candidates Output:     {total_cands:,} (Mean: {total_cands/max(1, total_s1):.1f} / S1)")
    print(f"  Total Matches Predicted:     {total_matches:,}")
    print(f"  Singletons Output:           {singletons:,} ({singletons/max(1, total_s1)*100:.1f}%)")
    print(f"  S2 Peak RSS:                 {s2_peak_rss:.1f} MB (Post-Cleanup: {s2_post_rss:.1f} MB)")
    print(f"  S3 Peak RSS:                 {s3_peak_rss:.1f} MB (Post-Cleanup: {s3_post_rss:.1f} MB)")
    print(f"  Matching TSV Size:           {round(final_matching_tsv.stat().st_size / (1024**2), 2)} MB")
    print(f"  Candidate TSV Size:          {round(final_candidate_tsv.stat().st_size / (1024**2), 2)} MB")
    print("=" * 90 + "\n")

    # 7. Run Official Submission Validator
    print(f"[VALIDATION] Running Official Submission Validator...")
    val_script = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    validator_passed = False
    
    if val_script.exists() and s1_file is not None:
        res = subprocess.run([
            sys.executable, str(val_script),
            "--matching", str(final_matching_tsv),
            "--candidate", str(final_candidate_tsv),
            "--test-dir", str(s1_file.parent)
        ], capture_output=True, text=True)
        
        print(f"  Validator Return Code: {res.returncode}")
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
        "total_s1": total_s1,
        "total_candidates": total_cands,
        "total_matches": total_matches,
        "singletons": singletons,
        "runtime_seconds": total_pipeline_time,
        "s2_peak_rss": s2_peak_rss,
        "s2_post_rss": s2_post_rss,
        "s3_peak_rss": s3_peak_rss,
        "s3_post_rss": s3_post_rss,
        "matching_tsv": str(final_matching_tsv),
        "candidate_tsv": str(final_candidate_tsv)
    }


def main():
    parser = argparse.ArgumentParser(description="High-Performance Sequential Production Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--model-path", type=str, default=None, help="Trained model path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder for submission TSVs")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate capacity per query")
    parser.add_argument("--threshold", type=float, default=0.83, help="Decision threshold")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for S1 processing")
    parser.add_argument("--s1-limit", type=int, default=None, help="Limit S1 queries for test runs")
    parser.add_argument("--target-limit", type=int, default=None, help="Limit target records for test runs")
    parser.add_argument("--no-stage-local", action="store_true", help="Disable staging files to local fast disk")
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
        stage_local=not args.no_stage_local,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
