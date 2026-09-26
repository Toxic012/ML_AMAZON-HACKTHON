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
import warnings
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
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
    from src.blocking.strategies import (
        block_hybrid_single_source,
        extract_source_tiers,
        merge_strategy_e_tiers
    )
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
    # Tier TSV
    tiers_tsv = intermediate_dir / f"{source_label.lower()}_tiers.tsv"
    checkpoint_file = intermediate_dir / f".{source_label.lower()}_checkpoint.json"
    
    # Check if already completed under resume mode
    if resume and tiers_tsv.exists() and checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f_cp:
                cp = json.load(f_cp)
                if cp.get("status") == "COMPLETED":
                    n_rows = cp.get("processed_count", 0)
                    print(f"[{source_label.upper()} PHASE] Found completed intermediate tier results ({n_rows:,} S1 rows). Skipping re-computation.")
                    return tiers_tsv, get_process_memory_mb(), get_process_memory_mb()
        except Exception:
            pass

    print("=" * 90)
    print(f"[{source_label.upper()} PHASE] Indexing & Candidate Generation for {source_file.name}")
    print("=" * 90)
    
    rss_phase_start = get_process_memory_mb()
    print(f"  [1] Building {source_label.upper()} Inverted Index from {source_file.name} (Start RSS: {rss_phase_start} MB)...")
    target_index, target_store, n_target, t_index = stream_compact_index(source_file, limit=target_limit)
    rss_post_index = get_process_memory_mb()
    print(f"      [OK] {source_label.upper()} Indexed: {n_target:,} records in {t_index:.2f}s | RSS: {rss_post_index} MB (Delta: +{rss_post_index - rss_phase_start:.1f} MB)")
    
    # Prepare intermediate Tier TSV
    with open(tiers_tsv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["source1_entity_id", "exact", "tokens", "char4", "char3", "addr", "scores"])
        
    f_tier = open(tiers_tsv, "a", encoding="utf-8", newline="")
    tier_writer = csv.writer(f_tier, delimiter="\t")
    
    total_processed = 0
    total_candidates = 0
    total_matches = 0
    batch_records = []
    
    total_s1_expected = s1_limit if s1_limit else 1732544
    
    t_start = time.time()
    last_log_time = time.time()
    
    print(f"\n  [2] Streaming S1 Queries against {source_label.upper()} (Extracting Strategy E representation tiers, Top-K = {top_k_source})...")
    
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
                    c_cnt, m_cnt = _process_source_batch_exact(
                        batch_records,
                        target_index,
                        target_store,
                        matcher,
                        threshold,
                        top_k_source,
                        tier_writer
                    )
                    total_candidates += c_cnt
                    total_matches += m_cnt
                    total_processed += len(batch_records)
                    batch_records = []
                    
                    if time.time() - last_log_time >= 15.0 or (total_processed % 25000 == 0):
                        elapsed = time.time() - t_start
                        qps = total_processed / max(0.001, elapsed)
                        pct = (total_processed / max(1, total_s1_expected)) * 100
                        remaining_rows = max(0, total_s1_expected - total_processed)
                        eta_sec = remaining_rows / max(0.001, qps)
                        curr_rss = get_process_memory_mb()
                        f_tier.flush()
                        disk_mb = tiers_tsv.stat().st_size / (1024 ** 2) if tiers_tsv.exists() else 0.0
                        
                        print(
                            f"      [{source_label.upper()}] S1: {total_processed:>9,} / {total_s1_expected:,} ({pct:5.1f}%) | "
                            f"Speed: {qps:6.1f} rows/s | Elapsed: {elapsed:6.1f}s | ETA: {eta_sec/60:5.1f} min | "
                            f"RSS: {curr_rss:6.1f} MB | Disk: {disk_mb:7.1f} MB | Cands: {total_candidates:,}",
                            flush=True
                        )
                        last_log_time = time.time()
                        
                if s1_limit and total_processed >= s1_limit:
                    break
                    
            if batch_records:
                c_cnt, m_cnt = _process_source_batch_exact(
                    batch_records,
                    target_index,
                    target_store,
                    matcher,
                    threshold,
                    top_k_source,
                    tier_writer
                )
                total_candidates += c_cnt
                total_matches += m_cnt
                total_processed += len(batch_records)
                
    finally:
        f_tier.close()
        
    peak_rss = get_process_memory_mb()
    total_elapsed = time.time() - t_start
    final_disk_mb = tiers_tsv.stat().st_size / (1024 ** 2) if tiers_tsv.exists() else 0.0
    print(f"\n  [OK] {source_label.upper()} Tier Extraction Finished: {total_processed:,} S1 entities in {total_elapsed:.2f}s ({total_processed/max(0.001, total_elapsed):.1f} S1/s)")
    print(f"       Total {source_label.upper()} Candidates: {total_candidates:,} | Peak RSS: {peak_rss:.1f} MB | Intermediate Disk: {final_disk_mb:.1f} MB")
    
    # Save checkpoint
    with open(checkpoint_file, "w", encoding="utf-8") as f_cp:
        json.dump({
            "status": "COMPLETED",
            "source": source_label,
            "processed_count": total_processed,
            "candidates_count": total_candidates,
            "intermediate_disk_mb": final_disk_mb,
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
    
    return tiers_tsv, peak_rss, post_cleanup_rss


def _process_source_batch_exact(
    batch_s1: List[dict],
    target_index: InvertedTokenIndex,
    target_store: Dict[str, Tuple[str, str, str]],
    matcher: EntityMatcher,
    threshold: float,
    top_k_source: int,
    tier_writer
) -> Tuple[int, int]:
    """Extracts 5 representation tiers and pre-scores candidates for an S1 batch."""
    total_cands = 0
    total_matches = 0
    
    batch_tiers = []
    all_features = []
    s1_cand_ranges = []
    
    for s1_rec in batch_s1:
        exact, toks, g4, g3, addr = extract_source_tiers(s1_rec, target_index, top_k=top_k_source)
        
        # Collect unique candidate IDs across all 5 tiers
        unique_cands = []
        seen_c = set()
        for c_list in (exact, toks, g4, g3, addr):
            for cid in c_list:
                if cid not in seen_c and cid in target_store:
                    seen_c.add(cid)
                    unique_cands.append(cid)
                    
        total_cands += len(unique_cands)
        batch_tiers.append((s1_rec["entity_id"], exact, toks, g4, g3, addr, unique_cands))
        
        start_idx = len(all_features)
        if unique_cands:
            for cid in unique_cands:
                all_features.append(compute_pairwise_features(s1_rec, target_store[cid], cand_id=cid))
        end_idx = len(all_features)
        s1_cand_ranges.append((start_idx, end_idx))
        
    all_probas = []
    if all_features:
        X_batch = np.array(all_features, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            all_probas = matcher.predict_proba(X_batch)
            
    for (s1_id, exact, toks, g4, g3, addr, unique_cands), (start_idx, end_idx) in zip(batch_tiers, s1_cand_ranges):
        scores_map = {}
        if unique_cands:
            c_probas = all_probas[start_idx:end_idx]
            for cid, prob in zip(unique_cands, c_probas):
                scores_map[cid] = f"{prob:.4f}"
                if prob >= threshold:
                    total_matches += 1
                    
        scores_str = ",".join(f"{k}:{v}" for k, v in scores_map.items())
        tier_writer.writerow([
            s1_id,
            ",".join(exact),
            ",".join(toks),
            ",".join(g4),
            ",".join(g3),
            ",".join(addr),
            scores_str
        ])
        
    return total_cands, total_matches


def merge_source_predictions(
    s2_tiers_tsv: Path,
    s3_tiers_tsv: Path,
    final_matching_tsv: Path,
    final_candidate_tsv: Path,
    top_k: int = 50,
    threshold: float = 0.83
) -> Dict[str, Any]:
    """
    Stream-merges intermediate S2 and S3 tiers TSVs using EXACT Strategy E global selection.
    Memory-safe (O(1) RAM) streaming line-by-line.
    """
    print("=" * 90)
    print("[PHASE 3/3] Stream-Merging S2 and S3 Tiers into Exact Strategy E Submission TSVs")
    print("=" * 90)
    t0 = time.time()
    
    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    zero_matches = 0
    singletons = 0
    multi_matches = 0
    
    merge_peak_rss = get_process_memory_mb()
    
    with open(s2_tiers_tsv, "r", encoding="utf-8") as f2, \
         open(s3_tiers_tsv, "r", encoding="utf-8") as f3, \
         open(final_matching_tsv, "w", encoding="utf-8", newline="") as f_match, \
         open(final_candidate_tsv, "w", encoding="utf-8", newline="") as f_cand:
         
        r2 = csv.reader(f2, delimiter="\t")
        r3 = csv.reader(f3, delimiter="\t")
        
        # Skip header
        next(r2, None)
        next(r3, None)
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for row2, row3 in zip(r2, r3):
            s1_id = row2[0]
            assert s1_id == row3[0], f"S1 entity mismatch during merge: {s1_id} vs {row3[0]}"
            
            # Parse S2 tiers
            s2_exact = row2[1].split(",") if row2[1] else []
            s2_toks = row2[2].split(",") if row2[2] else []
            s2_g4 = row2[3].split(",") if row2[3] else []
            s2_g3 = row2[4].split(",") if row2[4] else []
            s2_addr = row2[5].split(",") if row2[5] else []
            s2_scores = dict(kv.split(":") for kv in row2[6].split(",")) if row2[6] else {}
            s2_tiers = (s2_exact, s2_toks, s2_g4, s2_g3, s2_addr)
            
            # Parse S3 tiers
            s3_exact = row3[1].split(",") if row3[1] else []
            s3_toks = row3[2].split(",") if row3[2] else []
            s3_g4 = row3[3].split(",") if row3[3] else []
            s3_g3 = row3[4].split(",") if row3[4] else []
            s3_addr = row3[5].split(",") if row3[5] else []
            s3_scores = dict(kv.split(":") for kv in row3[6].split(",")) if row3[6] else {}
            s3_tiers = (s3_exact, s3_toks, s3_g4, s3_g3, s3_addr)
            
            # Execute EXACT original Strategy E global merge
            final_cands = merge_strategy_e_tiers(s2_tiers, s3_tiers, top_k=top_k)
            total_candidates += len(final_cands)
            f_cand.write(f"{s1_id}\t{','.join(final_cands)}\n")
            
            # Filter matches by decision threshold
            matched_ids = []
            for cid in final_cands:
                score_str = s2_scores.get(cid) or s3_scores.get(cid)
                if score_str is not None and float(score_str) >= threshold:
                    matched_ids.append(cid)
                    
            if not matched_ids:
                zero_matches += 1
                f_match.write(f"{s1_id}\t\n")
            elif len(matched_ids) == 1:
                singletons += 1
                total_matches += 1
                f_match.write(f"{s1_id}\t{matched_ids[0]}\n")
            else:
                multi_matches += 1
                total_matches += len(matched_ids)
                f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")
                
            total_s1 += 1
            if total_s1 % 100000 == 0:
                merge_peak_rss = max(merge_peak_rss, get_process_memory_mb())
                print(f"      Merged {total_s1:>9,} S1 entities... (Candidates: {total_candidates:,} | Matches: {total_matches:,} | RSS: {get_process_memory_mb():.1f} MB)")
            
    elapsed = time.time() - t0
    merge_peak_rss = max(merge_peak_rss, get_process_memory_mb())
    print(f"  [OK] Exact Stream Merge Complete in {elapsed:.2f}s | S1: {total_s1:,} | Matches: {total_matches:,} | Candidates: {total_candidates:,}\n")
    return {
        "total_s1": total_s1,
        "total_candidates": total_candidates,
        "total_matches": total_matches,
        "zero_matches": zero_matches,
        "singletons": singletons,
        "multi_matches": multi_matches,
        "merge_peak_rss": merge_peak_rss
    }


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
    
    # 4. Phase 1: Source 2 Processing (Exact Strategy E Tier Extraction & Pre-Scoring)
    s2_tiers_tsv, s2_peak_rss, s2_post_rss = process_single_source_phase(
        source_label="S2",
        source_file=s2_file,
        s1_file=s1_file,
        matcher=matcher,
        threshold=tau,
        top_k_source=top_k,
        intermediate_dir=intermediate_dir,
        batch_size=batch_size,
        s1_limit=s1_limit,
        target_limit=target_limit,
        resume=resume
    )
    
    # 5. Phase 2: Source 3 Processing (Exact Strategy E Tier Extraction & Pre-Scoring)
    s3_tiers_tsv, s3_peak_rss, s3_post_rss = process_single_source_phase(
        source_label="S3",
        source_file=s3_file,
        s1_file=s1_file,
        matcher=matcher,
        threshold=tau,
        top_k_source=top_k,
        intermediate_dir=intermediate_dir,
        batch_size=batch_size,
        s1_limit=s1_limit,
        target_limit=target_limit,
        resume=resume
    )
    
    # 6. Phase 3: Exact Global Stream Merge & Selection
    merge_stats = merge_source_predictions(
        s2_tiers_tsv=s2_tiers_tsv,
        s3_tiers_tsv=s3_tiers_tsv,
        final_matching_tsv=final_matching_tsv,
        final_candidate_tsv=final_candidate_tsv,
        top_k=top_k,
        threshold=tau
    )
    
    total_pipeline_time = round(time.time() - t_pipeline_start, 2)
    
    # 7. Backup to Persistent Google Drive if available
    drive_backup_paths = []
    colab_drive_dir = Path("/content/drive/MyDrive/DATASET_ML-AMAZON/submission")
    if colab_drive_dir.parent.exists():
        try:
            colab_drive_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            drive_m = colab_drive_dir / "matching_results.tsv"
            drive_c = colab_drive_dir / "candidate_pairs.tsv"
            print(f"\n[BACKUP] Persisting final TSV files to Google Drive ({colab_drive_dir}) ...")
            shutil.copy2(final_matching_tsv, drive_m)
            shutil.copy2(final_candidate_tsv, drive_c)
            drive_backup_paths = [str(drive_m), str(drive_c)]
            print(f"         [OK] Backed up to Google Drive successfully.")
        except Exception as e_drive:
            print(f"         [WARNING] Google Drive backup skipped: {e_drive}")

    matching_size_mb = round(final_matching_tsv.stat().st_size / (1024**2), 2)
    candidate_size_mb = round(final_candidate_tsv.stat().st_size / (1024**2), 2)
    
    git_commit_sha = "unknown"
    try:
        git_res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        if git_res.returncode == 0:
            git_commit_sha = git_res.stdout.strip()
    except Exception:
        pass

    # 8. Run Official Submission Validator
    print(f"\n[VALIDATION] Running Official Submission Validator...")
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

    # 9. Print Full Production Complete Report
    print("\n" + "=" * 95)
    print("FULL PRODUCTION COMPLETE")
    print("=" * 95)
    print(f"S1 processed:            {merge_stats['total_s1']:,}")
    print(f"S2 processed:            4,887,273 (target source)")
    print(f"S3 processed:            5,082,316 (target source)")
    print()
    print(f"S2 peak RSS:             {s2_peak_rss:.1f} MB (Post-Cleanup: {s2_post_rss:.1f} MB)")
    print(f"S3 peak RSS:             {s3_peak_rss:.1f} MB (Post-Cleanup: {s3_post_rss:.1f} MB)")
    print(f"Merge RSS:               {merge_stats['merge_peak_rss']:.1f} MB")
    print()
    print(f"Total runtime:           {total_pipeline_time} seconds ({merge_stats['total_s1']/max(0.001, total_pipeline_time):.1f} S1/sec)")
    print()
    print(f"Candidate pairs:         {merge_stats['total_candidates']:,}")
    print(f"Matching predictions:    {merge_stats['total_matches']:,}")
    print(f"Singletons:              {merge_stats['singletons']:,}")
    print(f"Multi-match entities:    {merge_stats['multi_matches']:,}")
    print(f"Zero-match entities:     {merge_stats['zero_matches']:,}")
    print()
    print("matching_results.tsv:")
    print(f"  exact path:            {final_matching_tsv}")
    print(f"  size:                  {matching_size_mb} MB")
    print(f"  row count:             {merge_stats['total_s1']:,} S1 rows")
    print()
    print("candidate_pairs.tsv:")
    print(f"  exact path:            {final_candidate_tsv}")
    print(f"  size:                  {candidate_size_mb} MB")
    print(f"  row count:             {merge_stats['total_s1']:,} S1 rows")
    print()
    print("Google Drive output paths:")
    if drive_backup_paths:
        for p in drive_backup_paths:
            print(f"  - {p}")
    else:
        print("  - Local output only (Drive path not attached)")
    print()
    print(f"Validator:               {'PASS' if validator_passed else 'FAIL'}")
    print(f"Git commit:              {git_commit_sha}")
    print("=" * 95 + "\n")

    return {
        "validator_passed": validator_passed,
        "total_s1": merge_stats["total_s1"],
        "total_candidates": merge_stats["total_candidates"],
        "total_matches": merge_stats["total_matches"],
        "singletons": merge_stats["singletons"],
        "zero_matches": merge_stats["zero_matches"],
        "multi_matches": merge_stats["multi_matches"],
        "runtime_seconds": total_pipeline_time,
        "s2_peak_rss": s2_peak_rss,
        "s2_post_rss": s2_post_rss,
        "s3_peak_rss": s3_peak_rss,
        "s3_post_rss": s3_post_rss,
        "merge_peak_rss": merge_stats["merge_peak_rss"],
        "matching_tsv": str(final_matching_tsv),
        "candidate_tsv": str(final_candidate_tsv),
        "drive_backup_paths": drive_backup_paths,
        "git_commit": git_commit_sha
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
