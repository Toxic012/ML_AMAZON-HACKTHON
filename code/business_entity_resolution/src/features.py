import math
from difflib import SequenceMatcher
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np

FEATURE_NAMES = [
    # --- Name Features ---
    "name_exact_match",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_token_overlap_ratio",
    "name_char3_jaccard",
    "name_char4_jaccard",
    "name_seq_sim",
    "name_len_diff",
    "name_len_ratio",
    "name_token_count_diff",
    
    # --- Address Features ---
    "addr_exact_match",
    "addr_token_jaccard",
    "addr_token_overlap_count",
    "addr_token_overlap_ratio",
    "addr_char3_jaccard",
    "addr_seq_sim",
    "postal_exact_match",
    "postal_overlap_count",
    "s1_addr_missing",
    "cand_addr_missing",
    "both_addr_missing",
    
    # --- Cross-field & Origin Features ---
    "country_match",
    "combined_token_jaccard",
    "is_source2",
    "is_source3"
]


def _jaccard(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    if inter == 0:
        return 0.0
    union = len(set1) + len(set2) - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    min_len = min(len(set1), len(set2))
    return inter / min_len if min_len > 0 else 0.0


def _seq_sim(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    len1 = len(s1)
    len2 = len(s2)
    # Fast character count matching 100% numerically identical to difflib.SequenceMatcher.quick_ratio()
    counts = {}
    for c in s2:
        counts[c] = counts.get(c, 0) + 1
    matches = 0
    for c in s1:
        n = counts.get(c, 0)
        if n > 0:
            matches += 1
            counts[c] = n - 1
    return 2.0 * matches / (len1 + len2)


try:
    from .normalization import extract_postal_tokens, char_ngrams, tokenize
except Exception:
    try:
        from src.normalization import extract_postal_tokens, char_ngrams, tokenize
    except Exception:
        from normalization import extract_postal_tokens, char_ngrams, tokenize


class PreprocessedCandidate:
    """
    Compact slotted object for cached candidate record representation.
    Avoids repetitive tokenization, n-gram extraction, and dictionary allocations.
    """
    __slots__ = (
        'norm_name', 'norm_addr', 'country', 'cid',
        'name_tokens', 'char_3grams', 'char_4grams',
        'addr_tokens', 'postal_tokens', 'comb_tokens',
        'len_name', 'is_s2', 'is_s3'
    )
    def __init__(self, raw_record: Any, cid: Optional[str] = None):
        if isinstance(raw_record, tuple):
            c_name = raw_record[0] or ""
            c_addr = raw_record[1] or ""
            c_country = (raw_record[2] or "").strip().upper()
            cand_id = cid or (raw_record[3] if len(raw_record) > 3 else "")
        elif isinstance(raw_record, dict):
            c_name = raw_record.get("norm_name", "")
            c_addr = raw_record.get("norm_addr", "")
            c_country = (raw_record.get("country", "") or "").strip().upper()
            cand_id = cid or raw_record.get("entity_id", "")
        else:
            c_name = getattr(raw_record, "norm_name", "")
            c_addr = getattr(raw_record, "norm_addr", "")
            c_country = getattr(raw_record, "country", "").strip().upper()
            cand_id = cid or getattr(raw_record, "entity_id", "")
            
        self.norm_name = c_name
        self.norm_addr = c_addr
        self.country = c_country
        self.cid = cand_id
        
        name_toks = c_name.split()
        self.name_tokens = set(name_toks)
        
        clean_name = "".join(name_toks)
        len_cn = len(clean_name)
        if len_cn <= 3:
            self.char_3grams = {clean_name} if clean_name else set()
        else:
            self.char_3grams = {clean_name[i:i+3] for i in range(len_cn - 2)}
            
        if len_cn <= 4:
            self.char_4grams = {clean_name} if clean_name else set()
        else:
            self.char_4grams = {clean_name[i:i+4] for i in range(len_cn - 3)}
            
        self.addr_tokens = set(c_addr.split())
        self.postal_tokens = extract_postal_tokens(c_addr)
        self.comb_tokens = self.name_tokens | self.addr_tokens
        self.len_name = len(c_name)
        self.is_s2 = 1.0 if (cand_id and cand_id.startswith("S2-")) else 0.0
        self.is_s3 = 1.0 if (cand_id and cand_id.startswith("S3-")) else 0.0


def compute_pairwise_features(s1_record: Dict[str, Any], cand_record: Any, cand_id: Optional[str] = None) -> List[float]:
    """
    Computes a 25-dimensional numeric feature vector for a candidate pair (s1, candidate).
    Vectorized and optimized for high-throughput inference (sub-millisecond per pair).
    Supports PreprocessedCandidate, compact tuples, and rich dicts.
    """
    # Name fields
    s1_name = s1_record.get("norm_name", "")
    s1_n_tok = s1_record.get("name_tokens", set())
    s1_c3 = s1_record.get("char_3grams", set())
    s1_c4 = s1_record.get("char_4grams", set())
    
    # Address fields
    s1_addr = s1_record.get("norm_addr", "")
    s1_a_tok = s1_record.get("addr_tokens", set())
    s1_postal = s1_record.get("postal_tokens", set())
    s1_country = s1_record.get("country", "").strip().upper()
    comb_s1 = s1_record.get("comb_tokens")
    if comb_s1 is None:
        comb_s1 = s1_n_tok | s1_a_tok

    if isinstance(cand_record, PreprocessedCandidate):
        c_name = cand_record.norm_name
        c_addr = cand_record.norm_addr
        c_country = cand_record.country
        c_n_tok = cand_record.name_tokens
        c_c3 = cand_record.char_3grams
        c_c4 = cand_record.char_4grams
        c_a_tok = cand_record.addr_tokens
        c_postal = cand_record.postal_tokens
        comb_c = cand_record.comb_tokens
        len_c = cand_record.len_name
        is_s2 = cand_record.is_s2
        is_s3 = cand_record.is_s3
    elif isinstance(cand_record, tuple):
        c_name = cand_record[0] or ""
        c_addr = cand_record[1] or ""
        c_country = (cand_record[2] or "").strip().upper()
        cid = cand_id or (cand_record[3] if len(cand_record) > 3 else "")
        
        c_n_tok = tokenize(c_name)
        c_c3 = char_ngrams(c_name, n=3)
        c_c4 = char_ngrams(c_name, n=4)
        c_a_tok = tokenize(c_addr)
        c_postal = extract_postal_tokens(c_addr)
        comb_c = c_n_tok | c_a_tok
        len_c = len(c_name)
        is_s2 = 1.0 if (cid and cid.startswith("S2-")) else 0.0
        is_s3 = 1.0 if (cid and cid.startswith("S3-")) else 0.0
    else:
        c_name = cand_record.get("norm_name", "")
        c_n_tok = cand_record.get("name_tokens", set())
        c_c3 = cand_record.get("char_3grams", set())
        c_c4 = cand_record.get("char_4grams", set())
        c_addr = cand_record.get("norm_addr", "")
        c_a_tok = cand_record.get("addr_tokens", set())
        c_postal = cand_record.get("postal_tokens", set())
        c_country = cand_record.get("country", "").strip().upper()
        cid = cand_id or cand_record.get("entity_id", "")
        comb_c = c_n_tok | c_a_tok
        len_c = len(c_name)
        is_s2 = 1.0 if (cid and cid.startswith("S2-")) else 0.0
        is_s3 = 1.0 if (cid and cid.startswith("S3-")) else 0.0
    
    # Name features
    name_exact = 1.0 if (s1_name and s1_name == c_name) else 0.0
    name_tok_jaccard = _jaccard(s1_n_tok, c_n_tok)
    name_tok_overlap = float(len(s1_n_tok & c_n_tok))
    name_tok_overlap_ratio = _overlap_ratio(s1_n_tok, c_n_tok)
    name_c3_jaccard = _jaccard(s1_c3, c_c3)
    name_c4_jaccard = _jaccard(s1_c4, c_c4)
    name_seq_sim = _seq_sim(s1_name, c_name)
    len_s1 = len(s1_name)
    name_len_diff = float(abs(len_s1 - len_c))
    name_len_ratio = (min(len_s1, len_c) / max(1, max(len_s1, len_c))) if (len_s1 > 0 and len_c > 0) else 0.0
    name_tok_count_diff = float(abs(len(s1_n_tok) - len(c_n_tok)))
    
    # Address features
    addr_exact = 1.0 if (s1_addr and s1_addr == c_addr) else 0.0
    addr_tok_jaccard = _jaccard(s1_a_tok, c_a_tok)
    addr_tok_overlap = float(len(s1_a_tok & c_a_tok))
    addr_tok_overlap_ratio = _overlap_ratio(s1_a_tok, c_a_tok)
    addr_c3_jaccard = _jaccard(s1_record.get("char_3grams_addr", set()), cand_record.get("char_3grams_addr", set())) if (isinstance(cand_record, dict) and "char_3grams_addr" in s1_record) else 0.0
    addr_seq_sim = _seq_sim(s1_addr, c_addr)
    postal_overlap_count = float(len(s1_postal & c_postal))
    postal_exact = 1.0 if postal_overlap_count > 0 else 0.0
    
    s1_addr_missing = 1.0 if not s1_addr else 0.0
    cand_addr_missing = 1.0 if not c_addr else 0.0
    both_addr_missing = 1.0 if (not s1_addr and not c_addr) else 0.0
    
    # Cross-field & Origin features
    country_match = 1.0 if (s1_country == c_country or not s1_country or not c_country) else 0.0
    combined_tok_jaccard = _jaccard(comb_s1, comb_c)
    
    return [
        name_exact,
        name_tok_jaccard,
        name_tok_overlap,
        name_tok_overlap_ratio,
        name_c3_jaccard,
        name_c4_jaccard,
        name_seq_sim,
        name_len_diff,
        name_len_ratio,
        name_tok_count_diff,
        
        addr_exact,
        addr_tok_jaccard,
        addr_tok_overlap,
        addr_tok_overlap_ratio,
        addr_c3_jaccard,
        addr_seq_sim,
        postal_exact,
        postal_overlap_count,
        s1_addr_missing,
        cand_addr_missing,
        both_addr_missing,
        
        country_match,
        combined_tok_jaccard,
        is_s2,
        is_s3
    ]


def compute_features(s1_record: Dict[str, Any], cand_record: Dict[str, Any]) -> List[float]:
    """Compatibility alias."""
    return compute_pairwise_features(s1_record, cand_record)
