def generate_candidates(s1_record, s2_index, s3_index, top_k=50):
    country = s1_record.get("country", "")
    tokens = s1_record.get("name_tokens", set())
    
    s2_cands = s2_index.search(country, tokens, min_overlap=1)
    s3_cands = s3_index.search(country, tokens, min_overlap=1)
    
    # Simple strategy: Jaccard overlap scoring for top K
    def score_cand(eid, index):
        cand_record = index.entity_store[eid]
        cand_tokens = cand_record.get("name_tokens", set())
        if not tokens or not cand_tokens:
            return 0
        intersection = len(tokens & cand_tokens)
        union = len(tokens | cand_tokens)
        return intersection / union if union > 0 else 0

    scored_cands = []
    for eid in s2_cands:
        scored_cands.append((eid, score_cand(eid, s2_index)))
    for eid in s3_cands:
        scored_cands.append((eid, score_cand(eid, s3_index)))
        
    # Sort by score desc
    scored_cands.sort(key=lambda x: x[1], reverse=True)
    
    return [eid for eid, score in scored_cands[:top_k]]
