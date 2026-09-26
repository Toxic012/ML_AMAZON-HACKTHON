# Phase 3 Report: Pairwise Feature Engineering & Supervised Matching

## 1. Executive Summary

In **Phase 3**, we implemented a high-throughput, 25-dimensional vectorized pairwise feature engineering pipeline and trained an entity-aware **LightGBM Matcher** optimized specifically for the official **per-S1 Macro $F_{0.5}$** objective function.

### Key Highlights:
- **Validation Macro $F_{0.5}$:** **98.11%** (Macro Precision: **99.32%**, Macro Recall: **96.38%**).
- **Validation Global Pairwise $F_{0.5}$:** **98.52%** (Precision: **99.14%**, Recall: **96.10%**).
- **Singleton Accuracy:** **100% (13 / 13)** correctly identified true singletons (no false matches generated).
- **Optimal Decision Threshold:** $\tau^* = 0.83$ (selected via fine-grained validation grid search over macro $F_{0.5}$).
- **Feature Extraction Throughput:** **8,866 candidate pairs/sec** (pure Python vectorized arithmetic without heavy dependencies).
- **Production Submission Formats:** Verified compliant with competition specifications (`matching_results.tsv` and `candidate_pairs.tsv`).

---

## 2. Pairwise Feature Engineering Architecture (25 Features)

All 25 features are computed in sub-milliseconds per candidate pair $(S_1, \text{Target})$:

### A. Name Similarity Features (10 features)
1. `name_exact_match`: Binary indicator for normalized exact name equality.
2. `name_token_jaccard`: Word token set Jaccard similarity.
3. `name_token_overlap_count`: Absolute count of overlapping non-stopword tokens.
4. `name_token_overlap_ratio`: Overlap normalized by minimum token set size.
5. `name_char3_jaccard`: Character 3-gram sub-word Jaccard similarity (Unicode-safe).
6. `name_char4_jaccard`: Character 4-gram sub-word Jaccard similarity.
7. `name_seq_sim`: SequenceMatcher fast normalized edit similarity ratio.
8. `name_len_diff`: Absolute difference in character lengths.
9. `name_len_ratio`: Ratio of shorter string length to longer string length.
10. `name_token_count_diff`: Absolute difference in token counts.

### B. Address Similarity Features (11 features)
11. `addr_exact_match`: Binary indicator for normalized address equality.
12. `addr_token_jaccard`: Address word token Jaccard similarity.
13. `addr_token_overlap_count`: Count of overlapping address tokens.
14. `addr_token_overlap_ratio`: Overlap normalized by minimum address token count.
15. `addr_char3_jaccard`: Address character 3-gram Jaccard similarity.
16. `addr_seq_sim`: SequenceMatcher address similarity ratio.
17. `postal_exact_match`: Binary indicator for postal / zip code overlap.
18. `postal_overlap_count`: Number of matching postal tokens / building numbers.
19. `s1_addr_missing`: Flag indicating if $S_1$ address is missing.
20. `cand_addr_missing`: Flag indicating if Target candidate address is missing.
21. `both_addr_missing`: Flag indicating if both addresses are missing.

### C. Cross-Field & Origin Features (4 features)
22. `country_match`: Binary indicator for matching country codes (or missing country).
23. `combined_token_jaccard`: Jaccard similarity over union of name and address tokens.
24. `is_source2`: Binary indicator if candidate originates from Source 2.
25. `is_source3`: Binary indicator if candidate originates from Source 3.

---

## 3. LightGBM Feature Importances

The top 10 most influential features learned by the LightGBM classifier:

| Rank | Feature | Importance (Split Count) | Rationale |
| :--- | :--- | :--- | :--- |
| **1** | `addr_seq_sim` | **1,294.0** | Precise address matching provides decisive confirmation |
| **2** | `name_seq_sim` | **1,285.0** | Edit similarity separates brand variations from distinct businesses |
| **3** | `combined_token_jaccard` | **1,228.0** | Joint entity semantic agreement across all text fields |
| **4** | `name_len_ratio` | **951.0** | Penalizes drastic length discrepancies |
| **5** | `name_char3_jaccard` | **944.0** | Sub-word resilience against spelling corruption |
| **6** | `addr_token_overlap_ratio` | **812.0** | Key locality / street overlap signal |
| **7** | `addr_token_jaccard` | **766.0** | Address similarity |
| **8** | `name_len_diff` | **640.0** | Length differential |
| **9** | `name_char4_jaccard` | **493.0** | 4-gram precision matching |
| **10** | `name_token_count_diff` | **416.0** | Structural token alignment |

---

## 4. Controlled Feature Ablation Study

| Ablation Configuration | Feature Count | Optimal $\tau^*$ | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Global Pairwise $F_{0.5}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Name Lexical Only** | 10 | 0.73 | 85.82% | 76.30% | **78.95%** | 80.13% |
| **2. Name + Address Lexical** | 21 | 0.85 | 99.45% | 95.98% | **98.08%** | 98.58% |
| **3. Full Feature Suite** | **25** | **0.83** | **99.32%** | **96.38%** | **98.11%** | **98.52%** |

*Insight:* Address features provide a massive +19.13% boost in Macro $F_{0.5}$ (from 78.95% to 98.08%), while cross-field and origin indicators provide the final calibration to achieve 98.11% macro $F_{0.5}$.

---

## 5. Decision Threshold Optimization

Grid search on validation set (300 S1 entities, 13,725 candidate pairs):
- At low thresholds ($\tau = 0.20$), recall is high (98.3%) but precision drops to 95.1% ($F_{0.5} = 95.2\%$).
- As threshold increases toward $\tau^* = 0.83$, false positives are virtually eliminated (Precision = **99.32%**), yielding the global peak **Macro $F_{0.5} = 98.11\%$**.

---

## 6. Submission File Verification

- `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids` (Comma-separated matches, empty for singletons).
- `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids` (All evaluated blocking candidates).
- Checked against `student_resource/utils/validate_submission.py`. Format, delimiters, row structures, and subset constraints verified 100% compliant.
