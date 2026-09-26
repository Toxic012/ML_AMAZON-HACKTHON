from typing import Dict, Set, List, Tuple
import numpy as np


def compute_s1_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (precision, recall, f0.5) for a single S1 entity.
    Strictly follows official entity resolution evaluation semantics:
    - Singleton true (no match) + Singleton pred (no match) -> P=1.0, R=1.0, F0.5=1.0
    - Singleton true (no match) + Non-empty pred -> P=0.0, R=1.0, F0.5=0.0
    - Non-empty true + Empty pred -> P=1.0, R=0.0, F0.5=0.0
    - Non-empty true + Non-empty pred -> Standard F0.5 calculation with beta=0.5 (precision weighted 2x recall).
    """
    len_true = len(true_matches)
    len_pred = len(pred_matches)
    
    if len_true == 0 and len_pred == 0:
        return 1.0, 1.0, 1.0
    elif len_true == 0 and len_pred > 0:
        return 0.0, 1.0, 0.0
    elif len_true > 0 and len_pred == 0:
        return 1.0, 0.0, 0.0
    
    overlap = len(true_matches & pred_matches)
    precision = overlap / len_pred
    recall = overlap / len_true
    
    beta_sq = 0.25 # 0.5^2
    denom = (beta_sq * precision) + recall
    if denom > 0:
        f05 = (1.0 + beta_sq) * (precision * recall) / denom
    else:
        f05 = 0.0
        
    return precision, recall, f05


def evaluate_predictions_macro_f05(
    ground_truth_map: Dict[str, Set[str]],
    predictions_map: Dict[str, Set[str]],
    s1_ids: List[str]
) -> Dict[str, float]:
    """
    Computes official macro-averaged metrics across all evaluated S1 entities.
    """
    precisions = []
    recalls = []
    f05s = []
    
    true_singletons = 0
    pred_singletons = 0
    correct_singletons = 0
    
    total_true_pairs = 0
    total_pred_pairs = 0
    total_true_positives = 0
    
    for s1_id in s1_ids:
        true_m = ground_truth_map.get(s1_id, set())
        pred_m = predictions_map.get(s1_id, set())
        
        p, r, f05 = compute_s1_entity_f05(true_m, pred_m)
        precisions.append(p)
        recalls.append(r)
        f05s.append(f05)
        
        if len(true_m) == 0:
            true_singletons += 1
        if len(pred_m) == 0:
            pred_singletons += 1
        if len(true_m) == 0 and len_pred_m == 0 if 'len_pred_m' in locals() else (len(true_m) == 0 and len(pred_m) == 0):
            correct_singletons += 1
            
        total_true_pairs += len(true_m)
        total_pred_pairs += len(pred_m)
        total_true_positives += len(true_m & pred_m)
        
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f05 = float(np.mean(f05s))
    
    # Global pairwise metrics
    global_p = total_true_positives / max(1, total_pred_pairs)
    global_r = total_true_positives / max(1, total_true_pairs)
    global_f05_denom = (0.25 * global_p) + global_r
    global_f05 = (1.25 * global_p * global_r / global_f05_denom) if global_f05_denom > 0 else 0.0
    
    return {
        "macro_f05": round(macro_f05, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "global_pairwise_precision": round(global_p, 6),
        "global_pairwise_recall": round(global_r, 6),
        "global_pairwise_f05": round(global_f05, 6),
        "total_evaluated_s1": len(s1_ids),
        "total_true_pairs": total_true_pairs,
        "total_predicted_pairs": total_pred_pairs,
        "total_true_positives": total_true_positives,
        "true_singletons": true_singletons,
        "predicted_singletons": pred_singletons,
        "correct_singletons": correct_singletons
    }
