import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import lightgbm as lgb

from src.features import FEATURE_NAMES, compute_pairwise_features
from src.matching.metrics import evaluate_predictions_macro_f05


class EntityMatcher:
    """
    Gradient Boosted Decision Tree Matcher for Pairwise Business Entity Resolution.
    Trained to predict probability P(match = 1 | s1_features, cand_features).
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        self.params = model_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names = list(FEATURE_NAMES)
        self.best_threshold: float = 0.50

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        # Calculate positive class weight to balance dataset
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = max(1.0, min(10.0, n_neg / max(1, n_pos))) if n_pos > 0 else 1.0
        
        fit_params = dict(self.params)
        fit_params["scale_pos_weight"] = scale_pos_weight
        
        self.model = lgb.LGBMClassifier(**fit_params)
        
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}

    def optimize_threshold(
        self,
        val_s1_ids: List[str],
        val_candidate_pairs: List[Tuple[str, str]], # List of (s1_id, cand_id)
        val_probas: np.ndarray,
        gt_map: Dict[str, Set[str]],
        threshold_grid: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Searches threshold grid to find optimal tau that maximizes official Macro F0.5.
        """
        if threshold_grid is None:
            threshold_grid = np.arange(0.15, 0.90, 0.02)
            
        # Group candidates by S1 ID
        s1_cands_map: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in val_s1_ids}
        for (s1_id, cand_id), prob in zip(val_candidate_pairs, val_probas):
            if s1_id in s1_cands_map:
                s1_cands_map[s1_id].append((cand_id, float(prob)))
                
        best_tau = 0.50
        best_f05 = -1.0
        best_metrics = {}
        grid_search_history = []
        
        for tau in threshold_grid:
            tau = round(float(tau), 4)
            preds_map: Dict[str, Set[str]] = {}
            for s1_id, cand_list in s1_cands_map.items():
                matched = {cid for cid, p in cand_list if p >= tau}
                preds_map[s1_id] = matched
                
            eval_res = evaluate_predictions_macro_f05(gt_map, preds_map, val_s1_ids)
            macro_f05 = eval_res["macro_f05"]
            
            grid_search_history.append({
                "threshold": tau,
                "macro_f05": macro_f05,
                "macro_precision": eval_res["macro_precision"],
                "macro_recall": eval_res["macro_recall"],
                "total_pred_pairs": eval_res["total_predicted_pairs"]
            })
            
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_tau = tau
                best_metrics = eval_res
                
        self.best_threshold = best_tau
        return best_tau, {
            "best_threshold": best_tau,
            "best_metrics": best_metrics,
            "grid_history": grid_search_history
        }

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "best_threshold": self.best_threshold,
                "params": self.params
            }, f)

    @classmethod
    def load(cls, filepath: Path) -> "EntityMatcher":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        matcher = cls(model_params=data.get("params"))
        matcher.model = data.get("model")
        matcher.feature_names = data.get("feature_names", list(FEATURE_NAMES))
        matcher.best_threshold = data.get("best_threshold", 0.50)
        return matcher
