import sys
import os
import csv
import time
import gc
import psutil
import subprocess
from pathlib import Path

REPO_ROOT = Path('.').resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.normalization import normalize_record, normalize_text
from src.blocking.token_index import InvertedTokenIndex
from src.blocking.strategies import block_hybrid_full_union
from src.matching.matcher import EntityMatcher
from src.features import compute_pairwise_features

def get_process_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)

s1_file = REPO_ROOT / 'student_resource' / 'dataset' / 'test' / 'test_source1.tsv'
s2_file = REPO_ROOT / 'student_resource' / 'dataset' / 'test' / 'test_source2.tsv'
s3_file = REPO_ROOT / 'student_resource' / 'dataset' / 'test' / 'test_source3.tsv'
model_path = REPO_ROOT / 'experiments' / 'PHASE_3_pairwise_matching' / 'matcher_model.pkl'

scratch_dir = REPO_ROOT / 'temp_exact_audit'
scratch_dir.mkdir(parents=True, exist_ok=True)
s2_inter_file = scratch_dir / 's2_tiers.tsv'
s3_inter_file = scratch_dir / 's3_tiers.tsv'

cand_out = scratch_dir / 'candidate_pairs.tsv'
match_out = scratch_dir / 'matching_results.tsv'

t0 = time.time()
peak_rss = 0.0

def update_peak_rss():
    global peak_rss
    peak_rss = max(peak_rss, get_process_rss_mb())

# --- Step 0: Load S1 Queries ---
print("Loading 2,000 S1 records...", flush=True)
s1_records = []
with open(s1_file, 'r', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
        if i >= 2000: break
        s1_records.append(normalize_record(row))
update_peak_rss()

# Create a slice test_source1.tsv in scratch_dir for validator to test against
s1_slice_path = scratch_dir / 'test_source1.tsv'
with open(s1_slice_path, 'w', encoding='utf-8', newline='') as f_s1_slice:
    writer = csv.writer(f_s1_slice, delimiter='\t')
    writer.writerow(['entity_id', 'business_name', 'business_address', 'country'])
    for s1 in s1_records:
        writer.writerow([s1['entity_id'], s1.get('norm_name', ''), s1.get('norm_addr', ''), s1.get('country', '')])

# Load Matcher
matcher = EntityMatcher.load(model_path)
tau = 0.83

# --- Helper functions for Strategy E Tiers ---
def extract_source_tiers(s1_record: dict, target_idx: InvertedTokenIndex, top_k: int = 50):
    country = s1_record.get('country', '')
    norm_name = s1_record.get('norm_name', '')
    tokens = s1_record.get('name_tokens', set())
    c3 = s1_record.get('char_3grams', set())
    c4 = s1_record.get('char_4grams', set())
    addr_tokens = s1_record.get('addr_tokens', set())
    postal_tokens = s1_record.get('postal_tokens', set())
    
    exact = target_idx.search_exact(country, norm_name)
    toks = target_idx.search_tokens(country, tokens, top_k=top_k)
    g4 = target_idx.search_char_ngrams(country, c4, n=4, top_k=top_k)
    g3 = target_idx.search_char_ngrams(country, c3, n=3, top_k=top_k)
    addr = target_idx.search_address(country, addr_tokens, postal_tokens, top_k=top_k)
    return (exact, toks, g4, g3, addr)

def merge_strategy_e_tiers(s2_tiers, s3_tiers, top_k=50):
    s2_exact, s2_tokens, s2_char4, s2_char3, s2_addr = s2_tiers
    s3_exact, s3_tokens, s3_char4, s3_char3, s3_addr = s3_tiers
    
    seen = set()
    result = []
    
    # 1. Exact Name Priority
    for eid in s2_exact + s3_exact:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
            
    # 2. Word Token Overlap
    rem = top_k - len(result)
    if rem > 0:
        quota = max(1, int(rem * 0.45))
        half_q = max(1, quota // 2)
        for eid in s2_tokens[:half_q] + s3_tokens[:half_q]:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    # 3. Char 4-Gram Overlap
    rem = top_k - len(result)
    if rem > 0:
        quota = max(1, int(rem * 0.35))
        half_q = max(1, quota // 2)
        for eid in s2_char4[:half_q] + s3_char4[:half_q]:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break

    # 4. Char 3-Gram Overlap
    rem = top_k - len(result)
    if rem > 0:
        quota = max(1, int(rem * 0.5))
        half_q = max(1, quota // 2)
        for eid in s2_char3[:half_q] + s3_char3[:half_q]:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    # 5. Address & Postal Token Overlap
    rem = top_k - len(result)
    if rem > 0:
        half_rem = max(1, rem // 2)
        for eid in s2_addr[:half_rem] + s3_addr[:half_rem]:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    return result[:top_k]

# =========================================================================
# PHASE 1: S2 INDEXING, TIER RETRIEVAL & SCORING
# =========================================================================
print("\n--- PHASE 1: S2 Indexing & Candidate Generation ---", flush=True)
s2_idx = InvertedTokenIndex()
s2_store = {}
with open(s2_file, 'r', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
        if i >= 50000: break
        eid = row['entity_id']
        n_name = normalize_text(row.get('business_name', ''))
        n_addr = normalize_text(row.get('business_address', ''))
        c = row.get('country', '').strip().upper()
        s2_idx.add_compact_record(eid, n_name, n_addr, c)
        s2_store[eid] = (n_name, n_addr, c)
update_peak_rss()
s2_peak_rss = get_process_rss_mb()
print(f"S2 Indexed (50k). RSS: {s2_peak_rss:.1f} MB", flush=True)

with open(s2_inter_file, 'w', encoding='utf-8', newline='') as out_f:
    writer = csv.writer(out_f, delimiter='\t')
    for s1 in s1_records:
        s1_id = s1['entity_id']
        exact, toks, g4, g3, addr = extract_source_tiers(s1, s2_idx, top_k=50)
        
        # Unique candidates across tiers to score
        unique_cands = []
        seen_c = set()
        for c_list in (exact, toks, g4, g3, addr):
            for cid in c_list:
                if cid not in seen_c and cid in s2_store:
                    seen_c.add(cid)
                    unique_cands.append(cid)
                    
        # Score candidates
        scores_map = {}
        if unique_cands:
            feats = [compute_pairwise_features(s1, s2_store[cid], cand_id=cid) for cid in unique_cands]
            probs = matcher.predict_proba(feats)
            for cid, p in zip(unique_cands, probs):
                scores_map[cid] = f"{p:.4f}"
                
        scores_str = ",".join(f"{k}:{v}" for k, v in scores_map.items())
        writer.writerow([
            s1_id,
            ",".join(exact),
            ",".join(toks),
            ",".join(g4),
            ",".join(g3),
            ",".join(addr),
            scores_str
        ])
update_peak_rss()

# Completely purge S2
del s2_idx
del s2_store
gc.collect()
s2_purged_rss = get_process_rss_mb()
print(f"S2 completely purged. Post-GC RSS: {s2_purged_rss:.1f} MB", flush=True)

# =========================================================================
# PHASE 2: S3 INDEXING, TIER RETRIEVAL & SCORING
# =========================================================================
print("\n--- PHASE 2: S3 Indexing & Candidate Generation ---", flush=True)
s3_idx = InvertedTokenIndex()
s3_store = {}
with open(s3_file, 'r', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
        if i >= 50000: break
        eid = row['entity_id']
        n_name = normalize_text(row.get('business_name', ''))
        n_addr = normalize_text(row.get('business_address', ''))
        c = row.get('country', '').strip().upper()
        s3_idx.add_compact_record(eid, n_name, n_addr, c)
        s3_store[eid] = (n_name, n_addr, c)
update_peak_rss()
s3_peak_rss = get_process_rss_mb()
print(f"S3 Indexed (50k). RSS: {s3_peak_rss:.1f} MB", flush=True)

with open(s3_inter_file, 'w', encoding='utf-8', newline='') as out_f:
    writer = csv.writer(out_f, delimiter='\t')
    for s1 in s1_records:
        s1_id = s1['entity_id']
        exact, toks, g4, g3, addr = extract_source_tiers(s1, s3_idx, top_k=50)
        
        # Unique candidates across tiers to score
        unique_cands = []
        seen_c = set()
        for c_list in (exact, toks, g4, g3, addr):
            for cid in c_list:
                if cid not in seen_c and cid in s3_store:
                    seen_c.add(cid)
                    unique_cands.append(cid)
                    
        # Score candidates
        scores_map = {}
        if unique_cands:
            feats = [compute_pairwise_features(s1, s3_store[cid], cand_id=cid) for cid in unique_cands]
            probs = matcher.predict_proba(feats)
            for cid, p in zip(unique_cands, probs):
                scores_map[cid] = f"{p:.4f}"
                
        scores_str = ",".join(f"{k}:{v}" for k, v in scores_map.items())
        writer.writerow([
            s1_id,
            ",".join(exact),
            ",".join(toks),
            ",".join(g4),
            ",".join(g3),
            ",".join(addr),
            scores_str
        ])
update_peak_rss()

# Completely purge S3
del s3_idx
del s3_store
gc.collect()
s3_purged_rss = get_process_rss_mb()
print(f"S3 completely purged. Post-GC RSS: {s3_purged_rss:.1f} MB", flush=True)

# =========================================================================
# PHASE 3: STREAM MERGE & EXACT GLOBAL K=50 SELECTION
# =========================================================================
print("\n--- PHASE 3: Stream Merge & Final TSV Generation ---", flush=True)

total_sequential_cands = 0
total_sequential_matches = 0

sequential_cands_by_s1 = {}
sequential_matches_by_s1 = {}

with open(s2_inter_file, 'r', encoding='utf-8') as f2, \
     open(s3_inter_file, 'r', encoding='utf-8') as f3, \
     open(cand_out, 'w', encoding='utf-8', newline='') as f_cand, \
     open(match_out, 'w', encoding='utf-8', newline='') as f_match:
     
    r2 = csv.reader(f2, delimiter='\t')
    r3 = csv.reader(f3, delimiter='\t')
    
    w_cand = csv.writer(f_cand, delimiter='\t')
    w_match = csv.writer(f_match, delimiter='\t')
    
    w_cand.writerow(['source1_entity_id', 'candidate_entity_ids'])
    w_match.writerow(['source1_entity_id', 'matched_entity_ids'])
    
    for row2, row3 in zip(r2, r3):
        s1_id = row2[0]
        assert s1_id == row3[0], f"S1 ID mismatch: {s1_id} vs {row3[0]}"
        
        # Parse S2 tiers
        s2_exact = row2[1].split(',') if row2[1] else []
        s2_toks = row2[2].split(',') if row2[2] else []
        s2_g4 = row2[3].split(',') if row2[3] else []
        s2_g3 = row2[4].split(',') if row2[4] else []
        s2_addr = row2[5].split(',') if row2[5] else []
        s2_scores = dict(kv.split(':') for kv in row2[6].split(',')) if row2[6] else {}
        s2_tiers = (s2_exact, s2_toks, s2_g4, s2_g3, s2_addr)
        
        # Parse S3 tiers
        s3_exact = row3[1].split(',') if row3[1] else []
        s3_toks = row3[2].split(',') if row3[2] else []
        s3_g4 = row3[3].split(',') if row3[3] else []
        s3_g3 = row3[4].split(',') if row3[4] else []
        s3_addr = row3[5].split(',') if row3[5] else []
        s3_scores = dict(kv.split(':') for kv in row3[6].split(',')) if row3[6] else {}
        s3_tiers = (s3_exact, s3_toks, s3_g4, s3_g3, s3_addr)
        
        # Execute EXACT original Strategy E global merge
        final_cands = merge_strategy_e_tiers(s2_tiers, s3_tiers, top_k=50)
        sequential_cands_by_s1[s1_id] = list(final_cands)
        total_sequential_cands += len(final_cands)
        
        matches = []
        for cid in final_cands:
            score_str = s2_scores.get(cid) or s3_scores.get(cid)
            if score_str is not None and float(score_str) >= tau:
                matches.append(cid)
                
        sequential_matches_by_s1[s1_id] = matches
        total_sequential_matches += len(matches)
        
        w_cand.writerow([s1_id, ','.join(final_cands)])
        w_match.writerow([s1_id, ','.join(matches)])

update_peak_rss()
total_runtime = time.time() - t0
inter_disk_mb = (s2_inter_file.stat().st_size + s3_inter_file.stat().st_size) / (1024 * 1024)

# =========================================================================
# BENCHMARK AGAINST ORIGINAL SIMULTANEOUS FULL UNION
# =========================================================================
print("\n--- BENCHMARK: Verifying Against Original Simultaneous Execution ---", flush=True)

# Rebuild simultaneous indexes for 100% verification
s2_idx_sim = InvertedTokenIndex()
s2_store_sim = {}
with open(s2_file, 'r', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
        if i >= 50000: break
        eid = row['entity_id']
        n_name = normalize_text(row.get('business_name', ''))
        n_addr = normalize_text(row.get('business_address', ''))
        c = row.get('country', '').strip().upper()
        s2_idx_sim.add_compact_record(eid, n_name, n_addr, c)
        s2_store_sim[eid] = (n_name, n_addr, c)

s3_idx_sim = InvertedTokenIndex()
s3_store_sim = {}
with open(s3_file, 'r', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
        if i >= 50000: break
        eid = row['entity_id']
        n_name = normalize_text(row.get('business_name', ''))
        n_addr = normalize_text(row.get('business_address', ''))
        c = row.get('country', '').strip().upper()
        s3_idx_sim.add_compact_record(eid, n_name, n_addr, c)
        s3_store_sim[eid] = (n_name, n_addr, c)

target_store_sim = {**s2_store_sim, **s3_store_sim}

exact_cand_matches = 0
exact_pred_matches = 0
sim_cands_total = 0
sim_matches_total = 0
cand_intersection = 0
cand_union = 0
changed_s1_decisions = 0

for s1 in s1_records:
    s1_id = s1['entity_id']
    sim_cands = block_hybrid_full_union(s1, s2_idx_sim, s3_idx_sim, top_k=50)
    sim_cands_total += len(sim_cands)
    
    seq_cands = sequential_cands_by_s1[s1_id]
    
    if sim_cands == seq_cands:
        exact_cand_matches += 1
        
    set_sim = set(sim_cands)
    set_seq = set(seq_cands)
    cand_intersection += len(set_sim & set_seq)
    cand_union += len(set_sim | set_seq)
    
    # Score simultaneous
    sim_matches = []
    if sim_cands:
        feats = [compute_pairwise_features(s1, target_store_sim[cid], cand_id=cid) for cid in sim_cands if cid in target_store_sim]
        probs = matcher.predict_proba(feats)
        sim_matches = [cid for cid, p in zip(sim_cands, probs) if p >= tau]
    sim_matches_total += len(sim_matches)
    
    seq_matches = sequential_matches_by_s1[s1_id]
    if sim_matches == seq_matches:
        exact_pred_matches += 1
    else:
        changed_s1_decisions += 1

# Run official submission validator
val_script = REPO_ROOT / 'student_resource' / 'utils' / 'validate_submission.py'
v_res = subprocess.run([
    sys.executable, str(val_script),
    '--matching', str(match_out),
    '--candidate', str(cand_out),
    '--test-dir', str(scratch_dir)
], capture_output=True, text=True)

print("\n" + "="*70)
print("FINAL EXACT SEMANTIC VERIFICATION REPORT (2,000 S1 Slice)")
print("="*70)
print(f"Total S1 Queries:                    {len(s1_records)}")
print(f"Candidate Set Exact Equality %:      {exact_cand_matches}/{len(s1_records)} ({exact_cand_matches/len(s1_records)*100:.2f}%)")
print(f"Candidate Jaccard Similarity:        {cand_intersection/cand_union*100:.2f}%")
print(f"Candidate Overlap %:                 {cand_intersection/sim_cands_total*100:.2f}%")
print(f"Total Simultaneous Candidates:       {sim_cands_total}")
print(f"Total Sequential Candidates:         {total_sequential_cands}")
print(f"Prediction Set Exact Equality %:     {exact_pred_matches}/{len(s1_records)} ({exact_pred_matches/len(s1_records)*100:.2f}%)")
print(f"Total Simultaneous Match Decisions:  {sim_matches_total}")
print(f"Total Sequential Match Decisions:    {total_sequential_matches}")
print(f"Changed S1 Decisions:                {changed_s1_decisions}")
print(f"Submission Validator Result:         {'PASSED' if v_res.returncode == 0 else 'FAILED'}")
print(f"Peak Process RSS:                    {peak_rss:.1f} MB")
print(f"Intermediate Disk Usage:             {inter_disk_mb:.2f} MB")
print(f"Total Execution Runtime:             {total_runtime:.2f} s")
print("="*70)
print(v_res.stdout.strip())
