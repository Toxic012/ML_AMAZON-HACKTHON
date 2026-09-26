#!/usr/bin/env python3
"""
S2 Indexing Diagnostic and Micro-Benchmark Suite
Amazon ML Challenge 2026 — Business Entity Resolution

Audits:
1. Sequential raw TSV reading throughput & memory stability (Test A-D).
2. S2 indexing logic scaling on controlled subsets (10k, 50k, 100k, 250k) (Test E).
3. Memory growth and throughput profiling.
"""

import os
import sys
import csv
import time
import psutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import resolve_dataset_paths, print_dataset_diagnostics
    from src.normalization import normalize_text, tokenize, char_ngrams, extract_postal_tokens
    from src.blocking.token_index import InvertedTokenIndex, BUSINESS_STOPWORDS, ADDRESS_STOPWORDS
except ImportError:
    from code.business_entity_resolution.src.config import resolve_dataset_paths, print_dataset_diagnostics
    from code.business_entity_resolution.src.normalization import normalize_text, tokenize, char_ngrams, extract_postal_tokens
    from code.business_entity_resolution.src.blocking.token_index import InvertedTokenIndex, BUSINESS_STOPWORDS, ADDRESS_STOPWORDS


def get_rss_mb() -> float:
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
    except Exception:
        return 0.0


def run_sequential_read_diagnostic(file_path: Path, limit: Optional[int] = None, log_every: int = 250000):
    """
    Test A-D: Pure sequential TSV stream reading without index construction.
    Measures I/O throughput, line counting, and verifies stream stability.
    """
    print("=" * 80)
    print(f"TEST A-D: SEQUENTIAL TSV STREAMING DIAGNOSTIC")
    print(f"File: {file_path}")
    print(f"File Size: {file_path.stat().st_size / (1024**2):.2f} MB")
    print("=" * 80)
    
    t0 = time.time()
    t_last = t0
    count = 0
    rss_start = get_rss_mb()
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            count += 1
            if count % log_every == 0:
                now = time.time()
                elapsed = now - t0
                step_elapsed = now - t_last
                step_rate = log_every / max(0.001, step_elapsed)
                avg_rate = count / max(0.001, elapsed)
                rss = get_rss_mb()
                print(f"[STREAM READ] rows={count:>9,} | elapsed={elapsed:6.1f}s | step_rate={step_rate:>8,.0f} rows/s | avg_rate={avg_rate:>8,.0f} rows/s | RSS={rss:6.1f} MB")
                t_last = now
            if limit and count >= limit:
                break
                
    total_elapsed = time.time() - t0
    rss_end = get_rss_mb()
    print("-" * 80)
    print(f"[STREAM READ COMPLETE] Processed {count:,} rows in {total_elapsed:.2f}s ({count/max(0.001, total_elapsed):,.0f} rows/s)")
    print(f"RSS Memory: Start={rss_start:.1f} MB -> End={rss_end:.1f} MB (Delta: {rss_end - rss_start:.1f} MB)")
    print("=" * 80 + "\n")


def run_controlled_subset_benchmarks(file_path: Path, subsets=[10000, 50000, 100000, 250000]):
    """
    Test E: Indexing benchmark across controlled subset sizes.
    """
    print("=" * 80)
    print(f"TEST E: S2 INDEXING SCALING BENCHMARK (CONTROLLED SUBSETS)")
    print(f"Target File: {file_path.name}")
    print("=" * 80)
    
    for limit in subsets:
        t0 = time.time()
        rss_start = get_rss_mb()
        index = InvertedTokenIndex()
        compact_store = {}
        count = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
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
                
                # Pass normalized data to index and store compact tuple
                index.add_compact_record(eid, norm_name, norm_addr, country)
                compact_store[eid] = (norm_name, norm_addr, country)
                
                count += 1
                if count >= limit:
                    break
                    
        elapsed = time.time() - t0
        rss_end = get_rss_mb()
        delta_rss = rss_end - rss_start
        rate = count / max(0.001, elapsed)
        
        print(f"  Subset N={count:>7,} | Time={elapsed:6.2f}s | Speed={rate:>7,.0f} rows/s | RSS Start={rss_start:5.1f}MB -> End={rss_end:5.1f}MB (Delta: +{delta_rss:5.1f}MB)")
        
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="S2 Indexing Diagnostic")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for stream read")
    parser.add_argument("--log-every", type=int, default=250000, help="Logging frequency")
    args = parser.parse_args()
    
    paths = resolve_dataset_paths(args.data_dir)
    print_dataset_diagnostics(paths)
    
    s2_file = paths.get("test_source2")
    assert s2_file and s2_file.exists(), f"Source 2 file missing: {s2_file}"
    
    # 1. Run Sequential Read Test A-D
    run_sequential_read_diagnostic(s2_file, limit=args.limit, log_every=args.log_every)
    
    # 2. Run Controlled Subset Test E
    run_controlled_subset_benchmarks(s2_file)


if __name__ == "__main__":
    main()
