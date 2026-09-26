from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

BUSINESS_STOPWORDS = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "services", "service", "solutions", "solution",
    "group", "enterprises", "enterprise", "holdings", "holding",
    "pvt", "private", "the", "and", "of", "in", "for", "technologies",
    "technology", "tech", "global", "international", "usa", "us", "india"
}

class InvertedTokenIndex:
    """
    Memory-efficient, country-partitioned token inverted index with frequency-based pruning
    and open-set fallback support.
    """
    def __init__(self, max_token_freq: int = 5000):
        self.max_token_freq = max_token_freq
        
        # country -> token -> list of entity_ids
        self.country_token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # country -> exact_norm_name -> list of entity_ids
        self.country_exact_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Global open-set fallback indices for missing/unknown countries
        self.global_token_index: Dict[str, List[str]] = defaultdict(list)
        self.global_exact_index: Dict[str, List[str]] = defaultdict(list)
        
        # Token document frequencies across all records
        self.token_doc_freq: Dict[str, int] = defaultdict(int)
        self.total_records = 0

    def add_record(self, record: dict):
        eid = record.get("entity_id")
        if not eid:
            return
            
        country = record.get("country", "").strip().upper()
        norm_name = record.get("norm_name", "").strip()
        tokens = record.get("name_tokens", set())
        
        self.total_records += 1
        
        # Index exact normalized name
        if norm_name:
            if country:
                self.country_exact_index[country][norm_name].append(eid)
            self.global_exact_index[norm_name].append(eid)
            
        # Index non-stopword tokens
        for token in tokens:
            if token and len(token) > 1 and token not in BUSINESS_STOPWORDS:
                self.token_doc_freq[token] += 1
                if country:
                    self.country_token_index[country][token].append(eid)
                self.global_token_index[token].append(eid)

    def search_exact(self, country: str, norm_name: str) -> List[str]:
        if not norm_name:
            return []
        country = country.strip().upper() if country else ""
        if country and country in self.country_exact_index:
            cands = self.country_exact_index[country].get(norm_name, [])
            if cands:
                return list(cands)
        return list(self.global_exact_index.get(norm_name, []))

    def search_tokens(self, country: str, tokens: Set[str], min_overlap: int = 1, top_k: int = 50) -> List[str]:
        if not tokens:
            return []
            
        country = country.strip().upper() if country else ""
        token_map = self.country_token_index.get(country) if country in self.country_token_index else self.global_token_index
        
        # Filter tokens by frequency threshold
        valid_tokens = [
            t for t in tokens
            if t in token_map and self.token_doc_freq.get(t, 0) <= self.max_token_freq and t not in BUSINESS_STOPWORDS
        ]
        
        if not valid_tokens:
            # Fallback to least frequent available tokens if all were filtered
            valid_tokens = sorted(
                [t for t in tokens if t in token_map and t not in BUSINESS_STOPWORDS],
                key=lambda x: self.token_doc_freq.get(x, float('inf'))
            )[:2]

        candidate_scores = defaultdict(float)
        for t in valid_tokens:
            doc_f = max(1, self.token_doc_freq.get(t, 1))
            # IDF-inspired token weight
            weight = 1.0 / (1.0 + 0.1 * doc_f)
            for eid in token_map[t]:
                candidate_scores[eid] += weight

        if not candidate_scores:
            return []

        # Sort candidates by accumulated score descending
        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, score in sorted_cands[:top_k]]
