#!/usr/bin/env python3
"""
Deep Diagnostic Profiler for Task 1 (Char3) and Task 2 (Candidate Preprocessing)
Analyzes posting list distributions, candidate caching dynamics, and time spent.
"""
import sys
import time
import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.normalization import normalize_record, tokenize, char_ngrams, extract_postal_tokens
from src.blocking.token_index import InvertedTokenIndex
from src.features import compute_pairwise_features, PreprocessedCandidate
from scripts.run_production_pipeline import stream_compact_index

def profile_char3_and_cache():
    s1_file = Path("student_resource/dataset/test/test_source1.tsv")
    s2_file = Path("student_resource/dataset/test/test_source2.tsv")
    
    print("Indexing FULL S2 (4,887,273 records)...")
    t0 = time.time()
    s2_index, s2_store, n_target, t_idx = stream_compact_index(s2_file)
    print(f"S2 Indexed in {time.time()-t0:.2f}s")
    
    # --- TASK 1: CHAR3 METRICS ---
    print("\n" + "=" * 80)
    print("TASK 1: CHAR3 INVERTED INDEX METRICS")
    print("=" * 80)
    
    # Posting list lengths
    lengths = []
    for c, gmap in s2_index.country_char3_index.items():
        for g, plist in gmap.items():
            lengths.append(len(plist))
    global_lengths = [len(plist) for plist in s2_index.global_char3_index.values()]
    
    all_lengths = np.array(lengths) if lengths else np.array(global_lengths)
    n_unique_char3 = len(s2_index.char3_doc_freq)
    total_postings = sum(s2_index.char3_doc_freq.values())
    
    print(f"Unique char3 grams in index:      {n_unique_char3:,}")
    print(f"Total char3 postings (doc_freq):  {total_postings:,}")
    print(f"Min posting length:               {np.min(all_lengths):,}")
    print(f"Median posting length:            {np.median(all_lengths):,.1f}")
    print(f"Mean posting length:              {np.mean(all_lengths):,.1f}")
    print(f"P90 posting length:               {np.percentile(all_lengths, 90):,.1f}")
    print(f"P95 posting length:               {np.percentile(all_lengths, 95):,.1f}")
    print(f"P99 posting length:               {np.percentile(all_lengths, 99):,.1f}")
    print(f"Max posting length:               {np.max(all_lengths):,}")
    
    # Load 2,000 S1 records
    s1_records = []
    with open(s1_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("entity_id"):
                continue
            s1_records.append(normalize_record(row))
            if len(s1_records) >= 2000:
                break
                
    # Inspect char3 search on 2000 S1 queries
    grams_queried = set()
    total_posting_ids_traversed = 0
    total_candidate_scores_accumulated = 0
    
    for s1 in s1_records:
        country = s1.get("country", "").strip().upper()
        c3 = s1.get("char_3grams", set())
        grams_queried.update(c3)
        
        ngram_map = s2_index.country_char3_index.get(country) if country in s2_index.country_char3_index else s2_index.global_char3_index
        doc_freq_map = s2_index.char3_doc_freq
        
        valid_ngrams = [
            g for g in c3
            if g in ngram_map and doc_freq_map.get(g, 0) <= s2_index.max_ngram_freq
        ]
        if not valid_ngrams:
            valid_ngrams = sorted(
                [g for g in c3 if g in ngram_map],
                key=lambda x: doc_freq_map.get(x, float('inf'))
            )[:4]
            
        for g in valid_ngrams:
            plist = ngram_map[g]
            total_posting_ids_traversed += len(plist)
            
    print(f"Unique char3 grams queried by 2k S1: {len(grams_queried):,}")
    print(f"Total posting IDs traversed in 2k S1: {total_posting_ids_traversed:,} ({total_posting_ids_traversed/len(s1_records):,.1f} per S1 query)")
    
    # --- TASK 2: CANDIDATE PREPROCESSING METRICS ---
    print("\n" + "=" * 80)
    print("TASK 2: CANDIDATE PREPROCESSING & CACHE METRICS")
    print("=" * 80)
    
    # Let's see how many unique candidates are generated per batch vs global
    all_retrieved_cids = []
    batch_unique_counts = []
    batch_size = 500
    
    for i in range(0, len(s1_records), batch_size):
        batch = s1_records[i:i+batch_size]
        batch_cids = set()
        for s1 in batch:
            exact = s2_index.search_exact(s1.get("country", ""), s1.get("norm_name", ""))
            toks = s2_index.search_tokens(s1.get("country", ""), s1.get("name_tokens", set()), top_k=50)
            g4 = s2_index.search_char_ngrams(s1.get("country", ""), s1.get("char_4grams", set()), n=4, top_k=50)
            g3 = s2_index.search_char_ngrams(s1.get("country", ""), s1.get("char_3grams", set()), n=3, top_k=50)
            addr = s2_index.search_address(s1.get("country", ""), s1.get("addr_tokens", set()), s1.get("postal_tokens", set()), top_k=50)
            
            seen = set()
            for cl in (exact, toks, g4, g3, addr):
                for cid in cl:
                    if cid not in seen and cid in s2_store:
                        seen.add(cid)
                        all_retrieved_cids.append(cid)
                        batch_cids.add(cid)
        batch_unique_counts.append(len(batch_cids))
        
    total_evaluations = len(all_retrieved_cids)
    global_unique_cids = len(set(all_retrieved_cids))
    print(f"Total candidate instances across 2k S1 queries: {total_evaluations:,}")
    print(f"Global unique candidate IDs across 2k S1 queries: {global_unique_cids:,}")
    print(f"Average unique candidate IDs per batch (size={batch_size}): {np.mean(batch_unique_counts):,.1f}")
    print(f"Repeated candidates across batches: {sum(batch_unique_counts) - global_unique_cids:,}")
    
    # Time spent constructing components of PreprocessedCandidate
    sample_cids = list(set(all_retrieved_cids))[:10000]
    t0 = time.perf_counter()
    for cid in sample_cids:
        raw = s2_store[cid]
        c_name = raw[0] or ""
        _ = tokenize(c_name)
    t_tok = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for cid in sample_cids:
        raw = s2_store[cid]
        c_name = raw[0] or ""
        _ = char_ngrams(c_name, n=3)
    t_c3 = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for cid in sample_cids:
        raw = s2_store[cid]
        c_name = raw[0] or ""
        _ = char_ngrams(c_name, n=4)
    t_c4 = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for cid in sample_cids:
        raw = s2_store[cid]
        c_addr = raw[1] or ""
        _ = extract_postal_tokens(c_addr)
    t_postal = time.perf_counter() - t0
    
    print(f"\nMicro-timing for 10,000 candidate objects construction:")
    print(f"  - tokenize(c_name):           {t_tok:8.4f} s")
    print(f"  - char_ngrams(c_name, n=3):   {t_c3:8.4f} s")
    print(f"  - char_ngrams(c_name, n=4):   {t_c4:8.4f} s")
    print(f"  - extract_postal_tokens:      {t_postal:8.4f} s")
    print(f"  - Total for 10k objects:      {t_tok+t_c3+t_c4+t_postal:8.4f} s")

if __name__ == "__main__":
    profile_char3_and_cache()
