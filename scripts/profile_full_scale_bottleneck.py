#!/usr/bin/env python3
"""
Full-Scale Inverted Index & Query Profiling Benchmark
Amazon ML Challenge 2026 — Business Entity Resolution

Benchmarks ONLY:
  - Full S2 Index (4,887,273 records)
  - Exactly 2,000 S1 queries

Breaks down runtime into:
  1. S2 Index Construction
  2. Candidate Retrieval:
     - Exact Name Lookup
     - Word Token Inverted Index
     - Character 4-gram Inverted Index
     - Character 3-gram Inverted Index
     - Address & Postal Inverted Index
  3. Strategy E Tier Deduplication
  4. Candidate Preprocessing & Batch Cache
  5. Pairwise Feature Extraction
  6. LightGBM Vectorized Scoring
  7. Intermediate TSV Serialization
"""
import os
import sys
import time
import csv
import json
import argparse
import psutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

# Increase csv field limit
csv.field_size_limit(sys.maxsize)

# Ensure proper sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    get_dataset_dir,
    resolve_dataset_paths,
    print_dataset_diagnostics,
    stage_test_files_locally,
    EXPERIMENTS_DIR,
    BASE_DIR
)
from src.normalization import normalize_record
from src.blocking.token_index import InvertedTokenIndex
from src.blocking.strategies import extract_source_tiers
from src.features import compute_pairwise_features, PreprocessedCandidate
from src.matching.matcher import EntityMatcher
from scripts.run_production_pipeline import stream_compact_index

def get_process_memory_mb() -> float:
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
    except Exception:
        return 0.0

def run_profiling(
    data_dir: Optional[str] = None,
    n_queries: int = 2000,
    batch_size: int = 500,
    top_k_source: int = 50,
    threshold: float = 0.83,
    model_path: Optional[str] = None
):
    print("=" * 90, flush=True)
    print("FULL-SCALE PRODUCTION INVERTED INDEX & PROFILING BENCHMARK", flush=True)
    print(f"Index Size: FULL S2 (~4,887,273 records) | Query Slice: Exactly {n_queries:,} S1 queries", flush=True)
    print("=" * 90, flush=True)

    # 1. Resolve Dataset Paths
    resolved = resolve_dataset_paths(data_dir)
    print(f"Data Directory: {resolved['data_dir']}", flush=True)
    
    staged = stage_test_files_locally(resolved)
    s1_file = staged["test_source1"]
    s2_file = staged["test_source2"]
    
    assert s1_file and s1_file.exists(), f"Missing S1 test file: {s1_file}"
    assert s2_file and s2_file.exists(), f"Missing S2 test file: {s2_file}"

    # 2. Load Trained Matcher Model
    if model_path is None:
        m_path = EXPERIMENTS_DIR / "PHASE_3_pairwise_matching" / "matcher_model.pkl"
    else:
        m_path = Path(model_path)
        
    print(f"\n[1/3] Loading Trained LightGBM Matcher from {m_path} ...", flush=True)
    matcher = EntityMatcher.load(m_path)
    tau = threshold if threshold is not None else matcher.best_threshold
    print(f"      Model Loaded successfully (tau* = {tau:.2f})", flush=True)

    # 3. Index FULL S2 Target Dataset
    rss_start = get_process_memory_mb()
    print(f"\n[2/3] Building FULL S2 Inverted Index from {s2_file.name} (Start RSS: {rss_start:.1f} MB)...", flush=True)
    t_idx_start = time.time()
    s2_index, s2_store, n_target, t_idx_time = stream_compact_index(s2_file)
    rss_post_index = get_process_memory_mb()
    print(
        f"      [OK] S2 Indexed: {n_target:,} records in {t_idx_time:.2f}s | "
        f"RSS: {rss_post_index:.1f} MB (Delta: +{rss_post_index - rss_start:.1f} MB)",
        flush=True
    )

    # 4. Stream exactly n_queries from S1
    print(f"\n[3/3] Loading exactly {n_queries:,} S1 Query Records from {s1_file.name} ...", flush=True)
    s1_records = []
    with open(s1_file, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t")
        for row in reader:
            if not row.get("entity_id"):
                continue
            s1_records.append(normalize_record(row))
            if len(s1_records) >= n_queries:
                break
                
    print(f"      Loaded {len(s1_records):,} normalized S1 queries. Starting fine-grained profiling...\n", flush=True)

    # Profiling Accumulators
    t_exact = 0.0
    t_tokens = 0.0
    t_char4 = 0.0
    t_char3 = 0.0
    t_addr = 0.0
    t_tier_dedup = 0.0
    t_cand_cache = 0.0
    t_feat_extract = 0.0
    t_lgbm = 0.0
    t_tsv = 0.0
    
    total_candidates_scored = 0
    total_matches = 0
    unique_cands_total = 0
    
    t_all_start = time.time()
    last_log_time = time.time()
    
    global_cand_cache = {}
    
    for i in range(0, len(s1_records), batch_size):
        batch_s1 = s1_records[i:i+batch_size]
        batch_tiers = []
        batch_unique_cids = set()
        
        for s1_rec in batch_s1:
            country = s1_rec.get("country", "")
            norm_name = s1_rec.get("norm_name", "")
            tokens = s1_rec.get("name_tokens", set())
            c3 = s1_rec.get("char_3grams", set())
            c4 = s1_rec.get("char_4grams", set())
            addr_tokens = s1_rec.get("addr_tokens", set())
            postal_tokens = s1_rec.get("postal_tokens", set())
            
            # Sub-component 1: Exact Name Search
            t0 = time.perf_counter()
            exact = s2_index.search_exact(country, norm_name)
            t1 = time.perf_counter()
            t_exact += (t1 - t0)
            
            # Sub-component 2: Word Token Search
            toks = s2_index.search_tokens(country, tokens, top_k=top_k_source)
            t2 = time.perf_counter()
            t_tokens += (t2 - t1)
            
            # Sub-component 3: Char 4-gram Search
            g4 = s2_index.search_char_ngrams(country, c4, n=4, top_k=top_k_source)
            t3 = time.perf_counter()
            t_char4 += (t3 - t2)
            
            # Sub-component 4: Char 3-gram Search
            g3 = s2_index.search_char_ngrams(country, c3, n=3, top_k=top_k_source)
            t4 = time.perf_counter()
            t_char3 += (t4 - t3)
            
            # Sub-component 5: Address/Postal Search
            addr = s2_index.search_address(country, addr_tokens, postal_tokens, top_k=top_k_source)
            t5 = time.perf_counter()
            t_addr += (t5 - t4)
            
            # Sub-component 6: Tier Deduplication
            unique_cands = []
            seen_c = set()
            for c_list in (exact, toks, g4, g3, addr):
                for cid in c_list:
                    if cid not in seen_c and cid in s2_store:
                        seen_c.add(cid)
                        unique_cands.append(cid)
            batch_unique_cids.update(unique_cands)
            batch_tiers.append((s1_rec["entity_id"], s1_rec, exact, toks, g4, g3, addr, unique_cands))
            t6 = time.perf_counter()
            t_tier_dedup += (t6 - t5)
            
        unique_cands_total += len(batch_unique_cids)
        
        # Sub-component 7: Candidate Preprocessing & Cache
        t_c0 = time.perf_counter()
        for cid in batch_unique_cids:
            if cid not in global_cand_cache:
                global_cand_cache[cid] = PreprocessedCandidate(s2_store[cid], cid)
        cand_cache = global_cand_cache
        t_c1 = time.perf_counter()
        t_cand_cache += (t_c1 - t_c0)
        
        # Sub-component 8: Pairwise Feature Extraction
        t_f0 = time.perf_counter()
        all_features = []
        s1_cand_ranges = []
        for s1_id, s1_rec, exact, toks, g4, g3, addr, unique_cands in batch_tiers:
            start_idx = len(all_features)
            if unique_cands:
                for cid in unique_cands:
                    all_features.append(compute_pairwise_features(s1_rec, cand_cache[cid]))
            end_idx = len(all_features)
            s1_cand_ranges.append((start_idx, end_idx))
        t_f1 = time.perf_counter()
        t_feat_extract += (t_f1 - t_f0)
        total_candidates_scored += len(all_features)
        
        # Sub-component 9: LightGBM Vectorized Inference
        t_m0 = time.perf_counter()
        all_probas = []
        if all_features:
            X_batch = np.array(all_features, dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                all_probas = matcher.predict_proba(X_batch)
        t_m1 = time.perf_counter()
        t_lgbm += (t_m1 - t_m0)
        
        # Sub-component 10: TSV Formatting
        t_w0 = time.perf_counter()
        for (s1_id, s1_rec, exact, toks, g4, g3, addr, unique_cands), (start_idx, end_idx) in zip(batch_tiers, s1_cand_ranges):
            scores_map = {}
            if unique_cands:
                c_probas = all_probas[start_idx:end_idx]
                for cid, prob in zip(unique_cands, c_probas):
                    scores_map[cid] = f"{prob:.4f}"
                    if prob >= threshold:
                        total_matches += 1
            scores_str = ",".join(f"{k}:{v}" for k, v in scores_map.items())
            _ = f"{s1_id}\t{','.join(exact)}\t{','.join(toks)}\t{','.join(g4)}\t{','.join(g3)}\t{','.join(addr)}\t{scores_str}"
        t_w1 = time.perf_counter()
        t_tsv += (t_w1 - t_w0)
        
        # Live progress output
        done = min(len(s1_records), i + batch_size)
        elapsed_now = time.time() - t_all_start
        qps_now = done / max(0.001, elapsed_now)
        rss_now = get_process_memory_mb()
        print(
            f"      [PROFILING PROGRESS] S1 Queries: {done:>6,} / {len(s1_records):,} ({done/len(s1_records)*100:5.1f}%) | "
            f"Speed: {qps_now:5.1f} S1/s | Elapsed: {elapsed_now:5.1f}s | RSS: {rss_now:6.1f} MB | "
            f"Candidates Scored: {total_candidates_scored:,}",
            flush=True
        )

    total_query_time = time.time() - t_all_start
    t_retrieval_total = t_exact + t_tokens + t_char4 + t_char3 + t_addr
    peak_rss = get_process_memory_mb()

    # ------------------------------------------------------------------
    # FINAL PROFILING REPORT
    # ------------------------------------------------------------------
    print("\n" + "=" * 90, flush=True)
    print("FULL-SCALE BOTTLENECK PROFILING REPORT (4,887,273 S2 INDEX vs 2,000 S1 QUERIES)", flush=True)
    print("=" * 90, flush=True)
    print(f"Total S2 Target Records Indexed:       {n_target:,}", flush=True)
    print(f"S2 Index Construction Time:            {t_idx_time:8.2f} s", flush=True)
    print(f"Total S1 Queries Evaluated:            {len(s1_records):,}", flush=True)
    print(f"Total Candidate Pairs Scored:          {total_candidates_scored:,}", flush=True)
    print(f"Total Matches Found (tau >= {threshold}):       {total_matches:,}", flush=True)
    print(f"Total Query Execution Time:            {total_query_time:8.2f} s", flush=True)
    print(f"Throughput:                            {len(s1_records)/total_query_time:8.2f} S1/sec", flush=True)
    print(f"Peak Process RSS:                      {peak_rss:8.1f} MB", flush=True)
    print("-" * 90, flush=True)
    print(f"{'STAGE / SUB-COMPONENT':<42} | {'TIME (s)':<12} | {'% OF TOTAL':<12} | {'PER QUERY (ms)':<15}", flush=True)
    print("-" * 90, flush=True)
    print(f"{'1. Candidate Retrieval (Total Index Search)':<42} | {t_retrieval_total:>10.3f} s | {(t_retrieval_total/total_query_time)*100:>10.1f}% | {(t_retrieval_total/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'   - 1a. Exact Name Lookup':<42} | {t_exact:>10.3f} s | {(t_exact/total_query_time)*100:>10.1f}% | {(t_exact/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'   - 1b. Word Token Inverted Index':<42} | {t_tokens:>10.3f} s | {(t_tokens/total_query_time)*100:>10.1f}% | {(t_tokens/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'   - 1c. Char 4-gram Inverted Index':<42} | {t_char4:>10.3f} s | {(t_char4/total_query_time)*100:>10.1f}% | {(t_char4/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'   - 1d. Char 3-gram Inverted Index':<42} | {t_char3:>10.3f} s | {(t_char3/total_query_time)*100:>10.1f}% | {(t_char3/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'   - 1e. Address & Postal Inverted Index':<42} | {t_addr:>10.3f} s | {(t_addr/total_query_time)*100:>10.1f}% | {(t_addr/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'2. Strategy E Tier Dedup & Union':<42} | {t_tier_dedup:>10.3f} s | {(t_tier_dedup/total_query_time)*100:>10.1f}% | {(t_tier_dedup/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'3. Candidate Preprocessing & Cache':<42} | {t_cand_cache:>10.3f} s | {(t_cand_cache/total_query_time)*100:>10.1f}% | {(t_cand_cache/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'4. Pairwise Feature Extraction (25 feats)':<42} | {t_feat_extract:>10.3f} s | {(t_feat_extract/total_query_time)*100:>10.1f}% | {(t_feat_extract/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'5. LightGBM Vectorized Scoring':<42} | {t_lgbm:>10.3f} s | {(t_lgbm/total_query_time)*100:>10.1f}% | {(t_lgbm/len(s1_records))*1000:>13.2f} ms", flush=True)
    print(f"{'6. TSV Serialization':<42} | {t_tsv:>10.3f} s | {(t_tsv/total_query_time)*100:>10.1f}% | {(t_tsv/len(s1_records))*1000:>13.2f} ms", flush=True)
    print("=" * 90, flush=True)

    # Comparison against 50k slice
    print("\nCOMPARISON: 50k-Target Slice vs. 4.887M Full-Scale Index:", flush=True)
    print(f"  - 50k Index Query Throughput:       ~61.6 - 73.5 S1/sec", flush=True)
    print(f"  - Full-Scale 4.887M Throughput:      {len(s1_records)/total_query_time:.2f} S1/sec", flush=True)
    print(f"  - Slowdown Factor:                  {(61.6 / (len(s1_records)/total_query_time)):.2f}x slower at full scale", flush=True)
    print("=" * 90 + "\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile full-scale S2 inverted index & query processing.")
    parser.add_argument("--data-dir", type=str, default=None, help="Root dataset directory path")
    parser.add_argument("--n-queries", type=int, default=2000, help="Number of S1 queries to profile (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=500, help="S1 batch size for profiling (default: 500)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K candidates per source (default: 50)")
    parser.add_argument("--threshold", type=float, default=0.83, help="Decision threshold (default: 0.83)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained matcher_model.pkl")
    args = parser.parse_args()
    
    run_profiling(
        data_dir=args.data_dir,
        n_queries=args.n_queries,
        batch_size=args.batch_size,
        top_k_source=args.top_k,
        threshold=args.threshold,
        model_path=args.model_path
    )
