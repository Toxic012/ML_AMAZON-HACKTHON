import sys
import csv
from pathlib import Path

REPO_ROOT = Path('.').resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.normalization import normalize_record, normalize_text
from src.blocking.token_index import InvertedTokenIndex
from src.blocking.strategies import block_hybrid_full_union, block_hybrid_single_source
from src.matching.matcher import EntityMatcher
from src.features import compute_pairwise_features

s1_file = REPO_ROOT / 'student_resource' / 'dataset' / 'test' / 'test_source1.tsv'
s2_file = REPO_ROOT / 'student_resource' / 'dataset' / 'test' / 'test_source2.tsv'
s3_file = REPO_ROOT / 'student_resource' / 'dataset' / 'test' / 'test_source3.tsv'
model_path = REPO_ROOT / 'experiments' / 'PHASE_3_pairwise_matching' / 'matcher_model.pkl'

print("Indexing 50k S2 records...", flush=True)
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

print("Indexing 50k S3 records...", flush=True)
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

target_store = {**s2_store, **s3_store}
matcher = EntityMatcher.load(model_path)
tau = 0.83

print("Loading 2,000 S1 records...", flush=True)
s1_records = []
with open(s1_file, 'r', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
        if i >= 2000: break
        s1_records.append(normalize_record(row))

print(f"Comparing candidate generation and predictions across {len(s1_records)} S1 entities...", flush=True)

cand_exact_matches = 0
pred_exact_matches = 0

total_cands_sim = 0
total_cands_seq = 0
cand_intersection_total = 0
cand_union_total = 0

total_preds_sim = 0
total_preds_seq = 0
pred_intersection_total = 0
pred_union_total = 0

changed_s1_cands = []
changed_s1_preds = []

jaccard_cand_list = []

for s1 in s1_records:
    s1_id = s1.get("entity_id", "")
    
    # 1. Simultaneous full union (K=50)
    cands_sim = block_hybrid_full_union(s1, s2_idx, s3_idx, top_k=50)
    set_sim = set(cands_sim)
    total_cands_sim += len(set_sim)
    
    # 2. Sequential (K=25 per source)
    cands_s2 = block_hybrid_single_source(s1, s2_idx, top_k=25)
    cands_s3 = block_hybrid_single_source(s1, s3_idx, top_k=25)
    cands_seq = cands_s2 + cands_s3
    set_seq = set(cands_seq)
    total_cands_seq += len(set_seq)
    
    # Intersection & Jaccard for Candidates
    inter_cands = set_sim & set_seq
    union_cands = set_sim | set_seq
    cand_intersection_total += len(inter_cands)
    cand_union_total += len(union_cands)
    jaccard_c = len(inter_cands) / len(union_cands) if union_cands else 1.0
    jaccard_cand_list.append(jaccard_c)
    
    if set_sim == set_seq:
        cand_exact_matches += 1
    else:
        changed_s1_cands.append({
            "s1_id": s1_id,
            "sim_count": len(set_sim),
            "seq_count": len(set_seq),
            "only_sim": set_sim - set_seq,
            "only_seq": set_seq - set_sim
        })
        
    # Model predictions on Simultaneous Candidates
    preds_sim = set()
    if cands_sim:
        feats_sim = [compute_pairwise_features(s1, target_store[cid], cand_id=cid) for cid in cands_sim if cid in target_store]
        if feats_sim:
            probs = matcher.predict_proba(feats_sim)
            for cid, p in zip([cid for cid in cands_sim if cid in target_store], probs):
                if p >= tau:
                    preds_sim.add(cid)
    total_preds_sim += len(preds_sim)
    
    # Model predictions on Sequential Candidates
    preds_seq = set()
    if cands_seq:
        feats_seq = [compute_pairwise_features(s1, target_store[cid], cand_id=cid) for cid in cands_seq if cid in target_store]
        if feats_seq:
            probs = matcher.predict_proba(feats_seq)
            for cid, p in zip([cid for cid in cands_seq if cid in target_store], probs):
                if p >= tau:
                    preds_seq.add(cid)
    total_preds_seq += len(preds_seq)
    
    inter_preds = preds_sim & preds_seq
    union_preds = preds_sim | preds_seq
    pred_intersection_total += len(inter_preds)
    pred_union_total += len(union_preds)
    
    if preds_sim == preds_seq:
        pred_exact_matches += 1
    else:
        changed_s1_preds.append({
            "s1_id": s1_id,
            "preds_sim": preds_sim,
            "preds_seq": preds_seq,
            "only_sim": preds_sim - preds_seq,
            "only_seq": preds_seq - preds_sim
        })

mean_jaccard_cands = sum(jaccard_cand_list) / len(jaccard_cand_list)
micro_jaccard_cands = cand_intersection_total / cand_union_total if cand_union_total > 0 else 1.0

print("\n" + "="*70)
print("DETERMINISTIC SEMANTIC COMPARISON RESULTS (2,000 S1 Queries)")
print("="*70)
print(f"Total S1 Queries Evaluated: {len(s1_records)}")
print(f"Candidate Set Exact Equality %:  {cand_exact_matches}/{len(s1_records)} ({cand_exact_matches/len(s1_records)*100:.2f}%)")
print(f"Candidate Mean Jaccard:          {mean_jaccard_cands:.4f} ({mean_jaccard_cands*100:.2f}%)")
print(f"Candidate Micro Jaccard:         {micro_jaccard_cands:.4f} ({micro_jaccard_cands*100:.2f}%)")
print(f"Total Candidates (Simultaneous): {total_cands_sim}")
print(f"Total Candidates (Sequential):   {total_cands_seq}")
print(f"Candidate Overlap Intersection:  {cand_intersection_total} ({(cand_intersection_total/total_cands_sim)*100:.2f}% of sim cands)")
print(f"Changed S1 Entities (Candidate): {len(changed_s1_cands)} / {len(s1_records)}")

print("\n" + "-"*70)
print(f"Prediction Set Exact Equality %: {pred_exact_matches}/{len(s1_records)} ({pred_exact_matches/len(s1_records)*100:.2f}%)")
print(f"Total Predicted Matches (Sim):   {total_preds_sim}")
print(f"Total Predicted Matches (Seq):   {total_preds_seq}")
print(f"Predicted Matches Intersection:  {pred_intersection_total}")
print(f"Matches Only in Simultaneous:    {total_preds_sim - pred_intersection_total}")
print(f"Matches Only in Sequential:      {total_preds_seq - pred_intersection_total}")
print(f"Changed S1 Entities (Match):     {len(changed_s1_preds)} / {len(s1_records)}")
print("="*70)

if changed_s1_preds:
    print("\nDetailed Inspection of Changed Prediction Entities:")
    for change in changed_s1_preds:
        print(f"  S1 ID: {change['s1_id']}")
        print(f"    Sim Preds: {change['preds_sim']}")
        print(f"    Seq Preds: {change['preds_seq']}")
        print(f"    Only Sim:  {change['only_sim']}")
        print(f"    Only Seq:  {change['only_seq']}")
