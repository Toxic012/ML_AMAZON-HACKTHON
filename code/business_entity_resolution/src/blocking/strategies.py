from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
from .token_index import InvertedTokenIndex, BUSINESS_STOPWORDS, ADDRESS_STOPWORDS


def block_exact_name(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """Strategy: Exact Name Key Blocking."""
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    
    s2_cands = s2_index.search_exact(country, norm_name)
    s3_cands = s3_index.search_exact(country, norm_name)
    
    seen = set()
    result = []
    for eid in s2_cands + s3_cands:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
        if len(result) >= top_k:
            break
    return result


def block_rare_tokens(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """Strategy: Rare Token Overlap Inverted Index Blocking."""
    country = s1_record.get("country", "")
    tokens = s1_record.get("name_tokens", set())
    
    half_k = max(1, top_k // 2)
    s2_cands = s2_index.search_tokens(country, tokens, top_k=half_k)
    s3_cands = s3_index.search_tokens(country, tokens, top_k=half_k)
    
    seen = set()
    result = []
    for eid in s2_cands + s3_cands:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
        if len(result) >= top_k:
            break
    return result


def block_char_3gram(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """Strategy: Character 3-gram Inverted Index Retrieval."""
    country = s1_record.get("country", "")
    c3 = s1_record.get("char_3grams", set())
    
    half_k = max(1, top_k // 2)
    s2_cands = s2_index.search_char_ngrams(country, c3, n=3, top_k=half_k)
    s3_cands = s3_index.search_char_ngrams(country, c3, n=3, top_k=half_k)
    
    seen = set()
    result = []
    for eid in s2_cands + s3_cands:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
        if len(result) >= top_k:
            break
    return result


def block_char_4gram(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """Strategy: Character 4-gram Inverted Index Retrieval."""
    country = s1_record.get("country", "")
    c4 = s1_record.get("char_4grams", set())
    
    half_k = max(1, top_k // 2)
    s2_cands = s2_index.search_char_ngrams(country, c4, n=4, top_k=half_k)
    s3_cands = s3_index.search_char_ngrams(country, c4, n=4, top_k=half_k)
    
    seen = set()
    result = []
    for eid in s2_cands + s3_cands:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
        if len(result) >= top_k:
            break
    return result


def block_address(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """Strategy: Address and Postal Token Retrieval."""
    country = s1_record.get("country", "")
    addr_tokens = s1_record.get("addr_tokens", set())
    postal_tokens = s1_record.get("postal_tokens", set())
    
    half_k = max(1, top_k // 2)
    s2_cands = s2_index.search_address(country, addr_tokens, postal_tokens, top_k=half_k)
    s3_cands = s3_index.search_address(country, addr_tokens, postal_tokens, top_k=half_k)
    
    seen = set()
    result = []
    for eid in s2_cands + s3_cands:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
        if len(result) >= top_k:
            break
    return result


# =========================================================================
# ABLATION SUITE (A, B, C, D, E)
# =========================================================================

def block_disjunctive_union(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """
    Ablation A: Disjunctive Union Baseline (Exact Name Priority + Rare Word Tokens).
    """
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    tokens = s1_record.get("name_tokens", set())
    
    # 1. Exact Name Priority
    s2_exact = s2_index.search_exact(country, norm_name)
    s3_exact = s3_index.search_exact(country, norm_name)
    
    seen = set()
    result = []
    for eid in s2_exact + s3_exact:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
            
    # 2. Token overlap matches fill remaining slots
    remaining = top_k - len(result)
    if remaining > 0:
        half_rem = max(1, remaining // 2)
        s2_tokens = s2_index.search_tokens(country, tokens, top_k=half_rem)
        s3_tokens = s3_index.search_tokens(country, tokens, top_k=half_rem)
        for eid in s2_tokens + s3_tokens:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    return result[:top_k]


def block_hybrid_A_char3(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """
    Ablation B: Disjunctive Union (A) + Character 3-Gram Overlap.
    """
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    tokens = s1_record.get("name_tokens", set())
    c3 = s1_record.get("char_3grams", set())
    
    seen = set()
    result = []
    
    # 1. Exact Name Priority
    for eid in s2_index.search_exact(country, norm_name) + s3_index.search_exact(country, norm_name):
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
            
    # 2. Token Overlap
    rem = top_k - len(result)
    if rem > 0:
        token_quota = max(1, int(rem * 0.6))
        half_q = max(1, token_quota // 2)
        for eid in s2_index.search_tokens(country, tokens, top_k=half_q) + s3_index.search_tokens(country, tokens, top_k=half_q):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    # 3. Char 3-Gram Overlap
    rem = top_k - len(result)
    if rem > 0:
        half_rem = max(1, rem // 2)
        for eid in s2_index.search_char_ngrams(country, c3, n=3, top_k=half_rem) + s3_index.search_char_ngrams(country, c3, n=3, top_k=half_rem):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    return result[:top_k]


def block_hybrid_A_char4(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """
    Ablation C: Disjunctive Union (A) + Character 4-Gram Overlap.
    """
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    tokens = s1_record.get("name_tokens", set())
    c4 = s1_record.get("char_4grams", set())
    
    seen = set()
    result = []
    
    # 1. Exact Name Priority
    for eid in s2_index.search_exact(country, norm_name) + s3_index.search_exact(country, norm_name):
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
            
    # 2. Token Overlap
    rem = top_k - len(result)
    if rem > 0:
        token_quota = max(1, int(rem * 0.6))
        half_q = max(1, token_quota // 2)
        for eid in s2_index.search_tokens(country, tokens, top_k=half_q) + s3_index.search_tokens(country, tokens, top_k=half_q):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    # 3. Char 4-Gram Overlap
    rem = top_k - len(result)
    if rem > 0:
        half_rem = max(1, rem // 2)
        for eid in s2_index.search_char_ngrams(country, c4, n=4, top_k=half_rem) + s3_index.search_char_ngrams(country, c4, n=4, top_k=half_rem):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    return result[:top_k]


def block_hybrid_A_char3_char4(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """
    Ablation D: Disjunctive Union (A) + Character 3-Gram + Character 4-Gram Overlap.
    """
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    tokens = s1_record.get("name_tokens", set())
    c3 = s1_record.get("char_3grams", set())
    c4 = s1_record.get("char_4grams", set())
    
    seen = set()
    result = []
    
    # 1. Exact Name Priority
    for eid in s2_index.search_exact(country, norm_name) + s3_index.search_exact(country, norm_name):
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
            
    # 2. Word Token Overlap (50% of remainder)
    rem = top_k - len(result)
    if rem > 0:
        quota = max(1, int(rem * 0.5))
        half_q = max(1, quota // 2)
        for eid in s2_index.search_tokens(country, tokens, top_k=half_q) + s3_index.search_tokens(country, tokens, top_k=half_q):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    # 3. Char 4-Gram Overlap (higher precision sub-word)
    rem = top_k - len(result)
    if rem > 0:
        quota = max(1, int(rem * 0.5))
        half_q = max(1, quota // 2)
        for eid in s2_index.search_char_ngrams(country, c4, n=4, top_k=half_q) + s3_index.search_char_ngrams(country, c4, n=4, top_k=half_q):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break

    # 4. Char 3-Gram Overlap (higher recall sub-word)
    rem = top_k - len(result)
    if rem > 0:
        half_rem = max(1, rem // 2)
        for eid in s2_index.search_char_ngrams(country, c3, n=3, top_k=half_rem) + s3_index.search_char_ngrams(country, c3, n=3, top_k=half_rem):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    return result[:top_k]


def block_hybrid_full_union(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """
    Ablation E: Hybrid Full Union (Ablation D + Address & Postal Token Retrieval).
    """
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    tokens = s1_record.get("name_tokens", set())
    c3 = s1_record.get("char_3grams", set())
    c4 = s1_record.get("char_4grams", set())
    addr_tokens = s1_record.get("addr_tokens", set())
    postal_tokens = s1_record.get("postal_tokens", set())
    
    seen = set()
    result = []
    
    # 1. Exact Name Priority
    for eid in s2_index.search_exact(country, norm_name) + s3_index.search_exact(country, norm_name):
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
            
    # 2. Word Token Overlap
    rem = top_k - len(result)
    if rem > 0:
        quota = max(1, int(rem * 0.45))
        half_q = max(1, quota // 2)
        for eid in s2_index.search_tokens(country, tokens, top_k=half_q) + s3_index.search_tokens(country, tokens, top_k=half_q):
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
        for eid in s2_index.search_char_ngrams(country, c4, n=4, top_k=half_q) + s3_index.search_char_ngrams(country, c4, n=4, top_k=half_q):
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
        for eid in s2_index.search_char_ngrams(country, c3, n=3, top_k=half_q) + s3_index.search_char_ngrams(country, c3, n=3, top_k=half_q):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    # 5. Address & Postal Token Overlap
    rem = top_k - len(result)
    if rem > 0:
        half_rem = max(1, rem // 2)
        for eid in s2_index.search_address(country, addr_tokens, postal_tokens, top_k=half_rem) + s3_index.search_address(country, addr_tokens, postal_tokens, top_k=half_rem):
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    return result[:top_k]


def block_composite_union(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """Legacy Composite Multi-Field Union Blocking."""
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    name_tokens = s1_record.get("name_tokens", set())
    addr_tokens = s1_record.get("addr_tokens", set())
    
    combined_tokens = set(name_tokens)
    for at in addr_tokens:
        if len(at) >= 4 and at not in BUSINESS_STOPWORDS:
            combined_tokens.add(at)

    s2_exact = s2_index.search_exact(country, norm_name)
    s3_exact = s3_index.search_exact(country, norm_name)
    
    seen = set()
    result = []
    for eid in s2_exact + s3_exact:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
            
    remaining = top_k - len(result)
    if remaining > 0:
        half_rem = max(1, remaining // 2)
        s2_cands = s2_index.search_tokens(country, combined_tokens, top_k=half_rem)
        s3_cands = s3_index.search_tokens(country, combined_tokens, top_k=half_rem)
        for eid in s2_cands + s3_cands:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
            if len(result) >= top_k:
                break
                
    return result[:top_k]


STRATEGIES = {
    # Baselines & Standalone
    "exact_name": block_exact_name,
    "rare_tokens": block_rare_tokens,
    "char_3gram": block_char_3gram,
    "char_4gram": block_char_4gram,
    "address_tokens": block_address,
    # Ablation Suite
    "ablation_A_disjunctive": block_disjunctive_union,
    "ablation_B_char3": block_hybrid_A_char3,
    "ablation_C_char4": block_hybrid_A_char4,
    "ablation_D_char3_char4": block_hybrid_A_char3_char4,
    "ablation_E_full_hybrid": block_hybrid_full_union,
    # Legacy alias
    "disjunctive_union": block_disjunctive_union,
    "composite_union": block_composite_union
}

