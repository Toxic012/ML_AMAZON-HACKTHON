def compute_features(s1_record, cand_record):
    s1_name_tokens = s1_record.get("name_tokens", set())
    cand_name_tokens = cand_record.get("name_tokens", set())
    
    s1_norm = s1_record.get("norm_name", "")
    cand_norm = cand_record.get("norm_name", "")
    exact_match = 1 if (s1_norm and s1_norm == cand_norm) else 0
    
    intersection = len(s1_name_tokens & cand_name_tokens)
    union = len(s1_name_tokens | cand_name_tokens)
    jaccard = intersection / union if union > 0 else 0
    
    same_country = 1 if (s1_record.get("country") == cand_record.get("country")) else 0
    
    return [exact_match, jaccard, same_country]
