#!/usr/bin/env python3
"""
Phase 3: Pairwise Feature Engineering, Gradient Boosted Matching & Threshold Optimization
Amazon ML Challenge 2026 — Business Entity Resolution

Workflow:
1. Load deterministic S1 queries & ground-truth mappings.
2. Index S2 & S3 target universe with guaranteed positive inclusion.
3. Generate candidate pairs using frozen EXP-0003 Full Hybrid Blocking (Strategy E).
4. Extract 25-dimensional vectorized pairwise features for all candidate pairs.
5. Entity-Aware Train/Val Split by S1 entity.
6. Train LightGBM EntityMatcher.
7. Optimize decision threshold specifically for official per-S1 Macro F0.5.
8. Run Feature Ablations.
9. Save Phase 3 experiment artifacts & trained model.
10. Generate submission files and run official validator.
"""

import os
import sys
import json
import time
import pickle
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# Setup sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.config import get_dataset_dir, BASE_DIR, EXPERIMENTS_DIR, SEED
    from src.data_loading import load_data_list
    from src.normalization import normalize_record
    from src.blocking.token_index import InvertedTokenIndex
    from src.blocking.strategies import block_hybrid_full_union
    from src.features import FEATURE_NAMES, compute_pairwise_features
    from src.matching.metrics import evaluate_predictions_macro_f05, compute_s1_entity_f05
    from src.matching.matcher import EntityMatcher
except ImportError:
    from code.business_entity_resolution.src.config import get_dataset_dir, BASE_DIR, EXPERIMENTS_DIR, SEED
    from code.business_entity_resolution.src.data_loading import load_data_list
    from code.business_entity_resolution.src.normalization import normalize_record
    from code.business_entity_resolution.src.blocking.token_index import InvertedTokenIndex
    from code.business_entity_resolution.src.blocking.strategies import block_hybrid_full_union
    from code.business_entity_resolution.src.features import FEATURE_NAMES, compute_pairwise_features
    from code.business_entity_resolution.src.matching.metrics import evaluate_predictions_macro_f05, compute_s1_entity_f05
    from code.business_entity_resolution.src.matching.matcher import EntityMatcher


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def index_and_cache_targets(source_file: Path, required_positive_ids: set, target_sample_size: int, max_token_freq: int = 5000):
    import csv
    csv.field_size_limit(sys.maxsize)
    
    index = InvertedTokenIndex(max_token_freq=max_token_freq)
    target_store = {}
    indexed_ids = set()
    
    found_positives = 0
    total_positives = len(required_positive_ids)
    negative_quota = max(0, target_sample_size - total_positives) if target_sample_size else float("inf")
    negative_count = 0
    
    t0 = time.time()
    with open(source_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id")
            if not eid:
                continue
                
            is_pos = eid in required_positive_ids
            is_neg = (not is_pos) and (negative_count < negative_quota)
            
            if is_pos or is_neg:
                norm_row = normalize_record(row)
                index.add_record(norm_row)
                target_store[eid] = norm_row
                indexed_ids.add(eid)
                if is_pos:
                    found_positives += 1
                else:
                    negative_count += 1
                    
            if target_sample_size and found_positives >= total_positives and negative_count >= negative_quota:
                break
                
    elapsed = round(time.time() - t0, 2)
    return index, target_store, indexed_ids, elapsed


def run_phase_3_pipeline(
    data_dir=None,
    s1_sample_size=2000,
    target_sample_size=30000,
    top_k=50,
    train_ratio=0.80,
    seed=42,
    experiment_id="PHASE_3_pairwise_matching"
):
    print("=" * 90)
    print(f"AMAZON ML CHALLENGE 2026 — {experiment_id}")
    print("PHASE 3: PAIRWISE FEATURE ENGINEERING & SUPERVISED MATCHING")
    print("=" * 90)
    
    random.seed(seed)
    np.random.seed(seed)
    resolved_data_dir = get_dataset_dir(data_dir)
    git_commit = get_git_commit()
    
    train_dir = resolved_data_dir / "train"
    gt_file = train_dir / "train_ground_truth.tsv"
    s1_file = train_dir / "train_source1.tsv"
    s2_file = train_dir / "train_source2.tsv"
    s3_file = train_dir / "train_source3.tsv"
    
    # 1. Load S1 Query records
    print(f"\n[1] Loading {s1_sample_size:,} Deterministic S1 Queries...")
    t0 = time.time()
    s1_records = load_data_list(s1_file, limit=s1_sample_size)
    s1_store = {r["entity_id"]: r for r in s1_records}
    s1_ids = [r["entity_id"] for r in s1_records]
    print(f"    Loaded {len(s1_records):,} S1 queries in {time.time() - t0:.2f}s")
    
    # 2. Extract Ground Truth
    print(f"\n[2] Extracting Ground Truth Target Mappings...")
    import csv
    gt_map = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            if s1_id in s1_store:
                m_str = row.get("matched_entity_ids", "")
                gt_map[s1_id] = {m.strip() for m in m_str.split(",") if m.strip()} if m_str else set()
                
    required_s2_ids = {m for matches in gt_map.values() for m in matches if m.startswith("S2-")}
    required_s3_ids = {m for matches in gt_map.values() for m in matches if m.startswith("S3-")}
    total_true_pairs = sum(len(m) for m in gt_map.values())
    print(f"    S1 Entities with Matches: {sum(1 for m in gt_map.values() if len(m) > 0):,} / {len(s1_records):,}")
    print(f"    Total True Match Pairs:   {total_true_pairs:,}")
    
    # 3. Index Target Sources S2 and S3
    print(f"\n[3] Indexing Target Universe with Guaranteed Positives...")
    s2_index, s2_store, indexed_s2, t_s2 = index_and_cache_targets(s2_file, required_s2_ids, target_sample_size)
    print(f"    [OK] S2 Universe: {len(indexed_s2):,} records in {t_s2}s")
    s3_index, s3_store, indexed_s3, t_s3 = index_and_cache_targets(s3_file, required_s3_ids, target_sample_size)
    print(f"    [OK] S3 Universe: {len(indexed_s3):,} records in {t_s3}s")
    
    target_store = {**s2_store, **s3_store}
    indexed_targets = indexed_s2 | indexed_s3
    
    # 4. Generate Candidate Pairs using Frozen Strategy E
    print(f"\n[4] Generating Candidate Pairs using Frozen Strategy E (top_k={top_k})...")
    t_cands = time.time()
    all_candidate_pairs = [] # (s1_id, cand_id, is_match)
    s1_candidates_map = {}
    
    recovered_positives = 0
    total_candidates_count = 0
    
    for s1 in s1_records:
        s1_id = s1["entity_id"]
        true_targets = gt_map.get(s1_id, set()) & indexed_targets
        
        cands = block_hybrid_full_union(s1, s2_index, s3_index, top_k=top_k)
        s1_candidates_map[s1_id] = cands
        total_candidates_count += len(cands)
        
        for cid in cands:
            is_match = 1 if cid in true_targets else 0
            if is_match:
                recovered_positives += 1
            all_candidate_pairs.append((s1_id, cid, is_match))
            
    cand_recall = recovered_positives / max(1, sum(len(gt_map.get(s["entity_id"], set()) & indexed_targets) for s in s1_records))
    print(f"    Generated {total_candidates_count:,} candidate pairs in {time.time() - t_cands:.2f}s")
    print(f"    Blocking Candidate True-Pair Recall: {cand_recall*100:.2f}% (Mean: {total_candidates_count/len(s1_records):.1f} cands/S1)")
    
    # 5. Extract Pairwise Features
    print(f"\n[5] Extracting 25-dim Vectorized Features for {len(all_candidate_pairs):,} Pairs...")
    t_feat = time.time()
    
    X_list = []
    y_list = []
    pair_meta_list = []
    
    for s1_id, cid, label in all_candidate_pairs:
        s1_rec = s1_store[s1_id]
        c_rec = target_store.get(cid)
        if c_rec is None:
            continue
        feats = compute_pairwise_features(s1_rec, c_rec)
        X_list.append(feats)
        y_list.append(label)
        pair_meta_list.append((s1_id, cid))
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    feat_time = time.time() - t_feat
    throughput = len(all_candidate_pairs) / max(0.001, feat_time)
    print(f"    Feature Extraction Complete in {feat_time:.2f}s ({throughput:.1f} pairs/sec)")
    print(f"    Feature Matrix Shape: {X.shape} | Positive Pairs: {int(np.sum(y)):,} | Negative Pairs: {int(len(y) - np.sum(y)):,}")

    # 6. Entity-Aware Train/Validation Split
    print(f"\n[6] Performing Entity-Aware S1 Train/Validation Split ({int(train_ratio*100)}/{int((1-train_ratio)*100)})...")
    shuffled_s1 = list(s1_ids)
    random.shuffle(shuffled_s1)
    split_idx = int(len(shuffled_s1) * train_ratio)
    train_s1_set = set(shuffled_s1[:split_idx])
    val_s1_set = set(shuffled_s1[split_idx:])
    val_s1_list = sorted(list(val_s1_set))
    
    train_mask = np.array([s1 in train_s1_set for s1, _ in pair_meta_list])
    val_mask = np.array([s1 in val_s1_set for s1, _ in pair_meta_list])
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pair_meta = [p for p, is_val in zip(pair_meta_list, val_mask) if is_val]
    
    print(f"    Train: {len(train_s1_set):,} S1 entities | {len(X_train):,} candidate pairs ({int(np.sum(y_train)):,} pos)")
    print(f"    Val:   {len(val_s1_set):,} S1 entities | {len(X_val):,} candidate pairs ({int(np.sum(y_val)):,} pos)")
    
    # 7. Train LightGBM Matcher
    print(f"\n[7] Training LightGBM Gradient Boosted Decision Tree Matcher...")
    t_train = time.time()
    matcher = EntityMatcher()
    matcher.fit(X_train, y_train, X_val, y_val)
    print(f"    Model Training Complete in {time.time() - t_train:.2f}s")
    
    # 8. Feature Importances
    importances = matcher.get_feature_importances()
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n    Top 10 Most Important Features:")
    for rank, (name, imp) in enumerate(top_features, 1):
        print(f"      {rank:2d}. {name:<28} : {imp:.1f}")

    # 9. Threshold Optimization on Validation Set
    print(f"\n[8] Optimizing Decision Threshold on Validation Set for Official Macro F0.5...")
    val_probas = matcher.predict_proba(X_val)
    best_tau, opt_info = matcher.optimize_threshold(
        val_s1_list,
        val_pair_meta,
        val_probas,
        gt_map
    )
    
    best_metrics = opt_info["best_metrics"]
    print(f"    [OPTIMAL THRESHOLD] tau* = {best_tau:.2f}")
    print(f"    -> Validation Macro F0.5:             {best_metrics['macro_f05']*100:.2f}%")
    print(f"    -> Validation Macro Precision:        {best_metrics['macro_precision']*100:.2f}%")
    print(f"    -> Validation Macro Recall:           {best_metrics['macro_recall']*100:.2f}%")
    print(f"    -> Validation Global Pairwise F0.5:   {best_metrics['global_pairwise_f05']*100:.2f}%")
    print(f"    -> Validation True Positives:         {best_metrics['total_true_positives']:,} / {best_metrics['total_true_pairs']:,}")
    print(f"    -> Correctly Identified Singletons:   {best_metrics['correct_singletons']:,} / {best_metrics['true_singletons']:,}")

    # 10. Feature Ablation Benchmark
    print(f"\n[9] Running Controlled Feature Ablations...")
    ablation_results = {}
    
    # Ablation 1: Name Lexical Only (first 10 features)
    X_tr_abl1, X_val_abl1 = X_train[:, :10], X_val[:, :10]
    m_abl1 = EntityMatcher()
    m_abl1.feature_names = FEATURE_NAMES[:10]
    m_abl1.fit(X_tr_abl1, y_train, X_val_abl1, y_val)
    tau1, opt1 = m_abl1.optimize_threshold(val_s1_list, val_pair_meta, m_abl1.predict_proba(X_val_abl1), gt_map)
    ablation_results["name_lexical_only"] = {
        "features": len(FEATURE_NAMES[:10]),
        "threshold": tau1,
        "metrics": opt1["best_metrics"]
    }
    print(f"    1. Name Lexical Only (10 feats)       : Macro F0.5 = {opt1['best_metrics']['macro_f05']*100:.2f}% (tau={tau1:.2f})")

    # Ablation 2: Name + Address Lexical (first 21 features)
    X_tr_abl2, X_val_abl2 = X_train[:, :21], X_val[:, :21]
    m_abl2 = EntityMatcher()
    m_abl2.feature_names = FEATURE_NAMES[:21]
    m_abl2.fit(X_tr_abl2, y_train, X_val_abl2, y_val)
    tau2, opt2 = m_abl2.optimize_threshold(val_s1_list, val_pair_meta, m_abl2.predict_proba(X_val_abl2), gt_map)
    ablation_results["name_and_address_lexical"] = {
        "features": len(FEATURE_NAMES[:21]),
        "threshold": tau2,
        "metrics": opt2["best_metrics"]
    }
    print(f"    2. Name + Address Lexical (21 feats)  : Macro F0.5 = {opt2['best_metrics']['macro_f05']*100:.2f}% (tau={tau2:.2f})")

    # Ablation 3: Full Feature Suite (all 25 features)
    ablation_results["full_feature_suite"] = {
        "features": len(FEATURE_NAMES),
        "threshold": best_tau,
        "metrics": best_metrics
    }
    print(f"    3. Full Feature Suite (25 feats)      : Macro F0.5 = {best_metrics['macro_f05']*100:.2f}% (tau={best_tau:.2f})")

    # 11. Save Production Artifacts
    exp_dir = EXPERIMENTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = exp_dir / "matcher_model.pkl"
    matcher.save(model_path)
    
    # Save feature schema
    schema_path = exp_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "top_features": top_features
        }, f, indent=2)
        
    # Save validation results
    val_res_path = exp_dir / "validation_results.json"
    with open(val_res_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_commit,
            "configuration": {
                "s1_sample_size": s1_sample_size,
                "target_sample_size": target_sample_size,
                "top_k": top_k,
                "train_s1_count": len(train_s1_set),
                "val_s1_count": len(val_s1_set),
                "total_candidate_pairs": len(all_candidate_pairs),
                "seed": seed
            },
            "best_threshold": best_tau,
            "validation_metrics": best_metrics,
            "ablations": ablation_results,
            "threshold_grid_history": opt_info["grid_history"]
        }, f, indent=2)
        
    print(f"\n[OK] Phase 3 Artifacts Saved to: {exp_dir}")

    # 12. Run End-to-End Production Submission Generator & Validator
    print(f"\n[10] Generating Production Submission TSVs & Validating...")
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"
    
    # Generate final matches on validation S1 set
    with open(matching_tsv, "w", encoding="utf-8") as f_match, open(candidate_tsv, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in val_s1_list:
            cands = s1_candidates_map.get(s1_id, [])
            cands_str = ",".join(cands)
            f_cand.write(f"{s1_id}\t{cands_str}\n")
            
            s1_rec = s1_store[s1_id]
            matched_ids = []
            for cid in cands:
                c_rec = target_store.get(cid)
                if c_rec:
                    feats = compute_pairwise_features(s1_rec, c_rec)
                    prob = float(matcher.predict_proba(np.array([feats]))[0])
                    if prob >= best_tau:
                        matched_ids.append(cid)
            matched_str = ",".join(matched_ids)
            f_match.write(f"{s1_id}\t{matched_str}\n")
            
    print(f"    Written: {matching_tsv}")
    print(f"    Written: {candidate_tsv}")
    
    # Run Submission Validation check
    val_script = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    if val_script.exists():
        print(f"    Running official validator: {val_script.name} ...")
        res = subprocess.run([
            sys.executable, str(val_script),
            "--matching", str(matching_tsv),
            "--candidate", str(candidate_tsv),
            "--test-dir", str(train_dir) # Use train dir containing train_source1.tsv for structure check
        ], capture_output=True, text=True)
        print(f"    Validator Return Code: {res.returncode}")
        print("    Validator Output Summary:")
        for line in res.stdout.strip().split("\n")[:10]:
            print(f"      {line}")
            
    return {
        "best_threshold": best_tau,
        "validation_metrics": best_metrics,
        "ablations": ablation_results,
        "matching_tsv": str(matching_tsv),
        "candidate_tsv": str(candidate_tsv)
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Pairwise Matching & Threshold Optimization")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--sample-size", type=int, default=1500, help="S1 query sample size")
    parser.add_argument("--target-sample-size", type=int, default=25000, help="Target sample size per source")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate capacity per query")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--experiment-id", type=str, default="PHASE_3_pairwise_matching", help="Experiment ID")
    
    args = parser.parse_args()
    
    run_phase_3_pipeline(
        data_dir=args.data_dir,
        s1_sample_size=args.sample_size,
        target_sample_size=args.target_sample_size,
        top_k=args.top_k,
        seed=args.seed,
        experiment_id=args.experiment_id
    )


if __name__ == "__main__":
    main()
