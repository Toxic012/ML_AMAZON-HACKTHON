from typing import List, Set, Dict, Tuple
from collections import defaultdict
from .token_index import InvertedTokenIndex, BUSINESS_STOPWORDS

def block_exact_name(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """Strategy 1: Exact Name Key Blocking."""
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    
    s2_cands = s2_index.search_exact(country, norm_name)
    s3_cands = s3_index.search_exact(country, norm_name)
    
    # Deduplicated list up to top_k
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
    """Strategy 2: Rare Token Overlap Inverted Index Blocking."""
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


def block_disjunctive_union(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """Strategy 3: Disjunctive Union Blocking (Exact Name + Rare Tokens)."""
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    tokens = s1_record.get("name_tokens", set())
    
    # 1. Exact matches have first priority
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


def block_composite_union(s1_record: dict, s2_index: InvertedTokenIndex, s3_index: InvertedTokenIndex, top_k: int = 50) -> List[str]:
    """
    Strategy 4: Composite Multi-Field Union Blocking (Exact Name + Name Tokens + Address Tokens).
    """
    country = s1_record.get("country", "")
    norm_name = s1_record.get("norm_name", "")
    name_tokens = s1_record.get("name_tokens", set())
    addr_tokens = s1_record.get("addr_tokens", set())
    
    # Combined token query with higher emphasis on name tokens
    combined_tokens = set(name_tokens)
    for at in addr_tokens:
        if len(at) >= 4 and at not in BUSINESS_STOPWORDS:
            combined_tokens.add(at)

    # 1. Exact Name Priority
    s2_exact = s2_index.search_exact(country, norm_name)
    s3_exact = s3_index.search_exact(country, norm_name)
    
    seen = set()
    result = []
    for eid in s2_exact + s3_exact:
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
            
    # 2. Multi-token / address expansion
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
    "exact_name": block_exact_name,
    "rare_tokens": block_rare_tokens,
    "disjunctive_union": block_disjunctive_union,
    "composite_union": block_composite_union
}
