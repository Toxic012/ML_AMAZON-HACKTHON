import sys
from pathlib import Path
import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.features import compute_pairwise_features, FEATURE_NAMES
    from src.normalization import normalize_record, normalize_text
except ImportError:
    from code.business_entity_resolution.src.features import compute_pairwise_features, FEATURE_NAMES
    from code.business_entity_resolution.src.normalization import normalize_record, normalize_text


def test_feature_count():
    assert len(FEATURE_NAMES) == 25


def test_features_dict_vs_tuple_equivalence():
    test_cases = [
        # (S1 row, Candidate row)
        (
            {"entity_id": "S1-100", "business_name": "Amazon Services LLC", "business_address": "410 Terry Ave N, Seattle, WA 98109", "country": "US"},
            {"entity_id": "S2-200", "business_name": "Amazon Web Services Inc", "business_address": "410 Terry Ave North, Seattle, 98109", "country": "US"}
        ),
        (
            {"entity_id": "S1-101", "business_name": "Café de la Paix", "business_address": "5 Place de l'Opéra, 75009 Paris", "country": "FR"},
            {"entity_id": "S3-301", "business_name": "Cafe de la Paix S.A.", "business_address": "5 Pl. de l'Opera, Paris 75009", "country": "FR"}
        ),
        (
            {"entity_id": "S1-102", "business_name": "Toyota Motor Corp", "business_address": "", "country": "JP"},
            {"entity_id": "S2-202", "business_name": "Honda Motor Co", "business_address": "Minato-ku Tokyo", "country": "JP"}
        ),
        (
            {"entity_id": "S1-103", "business_name": "Acme Corp", "business_address": "123 Main St", "country": ""},
            {"entity_id": "S3-303", "business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"}
        ),
        (
            {"entity_id": "S1-104", "business_name": "ООО Яндекс Маркет", "business_address": "ул. Льва Толстого, 16, Москва 119021", "country": "RU"},
            {"entity_id": "S2-204", "business_name": "Yandex Market LLC", "business_address": "Ul. Lva Tolstogo 16, Moscow", "country": "RU"}
        ),
    ]

    for s1_raw, cand_raw in test_cases:
        s1_norm = normalize_record(dict(s1_raw))
        
        # 1. Rich Dict evaluation
        cand_norm_dict = normalize_record(dict(cand_raw))
        feats_dict = compute_pairwise_features(s1_norm, cand_norm_dict)
        
        # 2. Compact Tuple evaluation
        c_norm_name = normalize_text(cand_raw["business_name"])
        c_norm_addr = normalize_text(cand_raw["business_address"])
        c_country = cand_raw["country"]
        cand_tuple = (c_norm_name, c_norm_addr, c_country)
        feats_tuple = compute_pairwise_features(s1_norm, cand_tuple, cand_id=cand_raw["entity_id"])
        
        assert len(feats_dict) == 25, f"Expected 25 features, got {len(feats_dict)}"
        assert len(feats_tuple) == 25, f"Expected 25 features, got {len(feats_tuple)}"
        
        for idx, (f_name, v_dict, v_tup) in enumerate(zip(FEATURE_NAMES, feats_dict, feats_tuple)):
            assert np.isclose(v_dict, v_tup, atol=1e-6), (
                f"Mismatch on feature '{f_name}' (idx {idx}): dict={v_dict} vs tuple={v_tup}\n"
                f"S1: {s1_raw}\nCand: {cand_raw}"
            )
