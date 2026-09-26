from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple, Any

BUSINESS_STOPWORDS = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "services", "service", "solutions", "solution",
    "group", "enterprises", "enterprise", "holdings", "holding",
    "pvt", "private", "the", "and", "of", "in", "for", "technologies",
    "technology", "tech", "global", "international", "usa", "us", "india"
}

ADDRESS_STOPWORDS = {
    "street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd",
    "lane", "ln", "drive", "dr", "court", "ct", "suite", "ste", "floor",
    "fl", "building", "bldg", "highway", "hwy", "box", "po", "usa", "us"
}

class InvertedTokenIndex:
    """
    Memory-efficient, country-partitioned multi-representation inverted index supporting:
    1. Exact normalized name index
    2. Rare word token inverted index with IDF weighting & frequency pruning
    3. Character 3-gram and 4-gram inverted index with sparse IDF overlap scoring
    4. Address & postal token inverted index
    5. Open-set fallback for missing/unmatched country codes.
    """
    def __init__(self, max_token_freq: int = 5000, max_ngram_freq: int = 5000):
        self.max_token_freq = max_token_freq
        self.max_ngram_freq = max_ngram_freq
        
        # Exact Name Index: country -> norm_name -> [entity_ids]
        self.country_exact_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.global_exact_index: Dict[str, List[str]] = defaultdict(list)
        
        # Word Token Index: country -> token -> [entity_ids]
        self.country_token_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.global_token_index: Dict[str, List[str]] = defaultdict(list)
        self.token_doc_freq: Dict[str, int] = defaultdict(int)
        
        # Character 3-gram Index: country -> 3gram -> [entity_ids]
        self.country_char3_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.global_char3_index: Dict[str, List[str]] = defaultdict(list)
        self.char3_doc_freq: Dict[str, int] = defaultdict(int)

        # Character 4-gram Index: country -> 4gram -> [entity_ids]
        self.country_char4_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.global_char4_index: Dict[str, List[str]] = defaultdict(list)
        self.char4_doc_freq: Dict[str, int] = defaultdict(int)

        # Address & Postal Index: country -> token -> [entity_ids]
        self.country_addr_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.global_addr_index: Dict[str, List[str]] = defaultdict(list)
        self.addr_doc_freq: Dict[str, int] = defaultdict(int)
        
        self.total_records = 0

    def add_record(self, record: dict):
        eid = record.get("entity_id")
        if not eid:
            return
            
        country = record.get("country", "").strip().upper()
        norm_name = record.get("norm_name", "").strip()
        tokens = record.get("name_tokens", set())
        char3s = record.get("char_3grams", set())
        char4s = record.get("char_4grams", set())
        addr_tokens = record.get("addr_tokens", set())
        postal_tokens = record.get("postal_tokens", set())
        
        self.total_records += 1
        
        # 1. Index exact normalized name
        if norm_name:
            if country:
                self.country_exact_index[country][norm_name].append(eid)
            self.global_exact_index[norm_name].append(eid)
            
        # 2. Index word tokens
        for token in tokens:
            if token and len(token) > 1 and token not in BUSINESS_STOPWORDS:
                self.token_doc_freq[token] += 1
                if country:
                    self.country_token_index[country][token].append(eid)
                self.global_token_index[token].append(eid)

        # 3. Index character 3-grams
        for g3 in char3s:
            if g3:
                self.char3_doc_freq[g3] += 1
                if country:
                    self.country_char3_index[country][g3].append(eid)
                self.global_char3_index[g3].append(eid)

        # 4. Index character 4-grams
        for g4 in char4s:
            if g4:
                self.char4_doc_freq[g4] += 1
                if country:
                    self.country_char4_index[country][g4].append(eid)
                self.global_char4_index[g4].append(eid)

        # 5. Index address & postal tokens
        all_addr = set(postal_tokens)
        for at in addr_tokens:
            if at and len(at) >= 3 and at not in ADDRESS_STOPWORDS:
                all_addr.add(at)
                
        for t in all_addr:
            self.addr_doc_freq[t] += 1
            if country:
                self.country_addr_index[country][t].append(eid)
            self.global_addr_index[t].append(eid)

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
            valid_tokens = sorted(
                [t for t in tokens if t in token_map and t not in BUSINESS_STOPWORDS],
                key=lambda x: self.token_doc_freq.get(x, float('inf'))
            )[:2]

        candidate_scores = defaultdict(float)
        for t in valid_tokens:
            doc_f = max(1, self.token_doc_freq.get(t, 1))
            weight = 1.0 / (1.0 + 0.1 * doc_f)
            for eid in token_map[t]:
                candidate_scores[eid] += weight

        if not candidate_scores:
            return []

        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, score in sorted_cands[:top_k]]

    def search_char_ngrams(self, country: str, ngrams: Set[str], n: int = 3, top_k: int = 50) -> List[str]:
        """
        Retrieves candidates based on character n-gram overlap with IDF-inspired weighting.
        """
        if not ngrams:
            return []
            
        country = country.strip().upper() if country else ""
        if n == 3:
            ngram_map = self.country_char3_index.get(country) if country in self.country_char3_index else self.global_char3_index
            doc_freq_map = self.char3_doc_freq
        else:
            ngram_map = self.country_char4_index.get(country) if country in self.country_char4_index else self.global_char4_index
            doc_freq_map = self.char4_doc_freq
            
        valid_ngrams = [
            g for g in ngrams
            if g in ngram_map and doc_freq_map.get(g, 0) <= self.max_ngram_freq
        ]
        
        if not valid_ngrams:
            valid_ngrams = sorted(
                [g for g in ngrams if g in ngram_map],
                key=lambda x: doc_freq_map.get(x, float('inf'))
            )[:4]
            
        candidate_scores = defaultdict(float)
        for g in valid_ngrams:
            df = max(1, doc_freq_map.get(g, 1))
            weight = 1.0 / (1.0 + 0.05 * df)
            for eid in ngram_map[g]:
                candidate_scores[eid] += weight
                
        if not candidate_scores:
            return []
            
        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, score in sorted_cands[:top_k]]

    def search_address(self, country: str, addr_tokens: Set[str], postal_tokens: Set[str], top_k: int = 50) -> List[str]:
        """
        Retrieves candidates based on address and postal code overlap.
        Postal code matches are given high initial weight.
        """
        if not addr_tokens and not postal_tokens:
            return []
            
        country = country.strip().upper() if country else ""
        addr_map = self.country_addr_index.get(country) if country in self.country_addr_index else self.global_addr_index
        
        candidate_scores = defaultdict(float)
        
        # High weight for postal codes (e.g. 3.0 per matching postal code)
        for pt in postal_tokens:
            if pt in addr_map:
                df = max(1, self.addr_doc_freq.get(pt, 1))
                weight = 3.0 / (1.0 + 0.05 * df)
                for eid in addr_map[pt]:
                    candidate_scores[eid] += weight
                    
        # Weight for general address tokens
        valid_addr_tokens = [
            at for at in addr_tokens
            if at in addr_map and self.addr_doc_freq.get(at, 0) <= self.max_token_freq and at not in ADDRESS_STOPWORDS
        ]
        
        for at in valid_addr_tokens:
            df = max(1, self.addr_doc_freq.get(at, 1))
            weight = 1.0 / (1.0 + 0.1 * df)
            for eid in addr_map[at]:
                candidate_scores[eid] += weight
                
        if not candidate_scores:
            return []
            
        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, score in sorted_cands[:top_k]]

