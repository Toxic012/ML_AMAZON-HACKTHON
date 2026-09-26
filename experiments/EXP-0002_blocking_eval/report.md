# Experiment Report: EXP-0002 Blocking Evaluation Methodology Audit & Corrected Benchmark

## 1. Executive Summary

An audit of the initial EXP-0002 blocking evaluation revealed that the reported low candidate recall (0.72% - 2.90%) was an artifact of a sampling omission flaw in the evaluation harness rather than algorithmic failure of the blocking strategies.

When evaluated against a properly constructed retrieval universe (guaranteeing target inclusion for sampled S1 queries alongside 38,000+ realistic negative distractors), the actual true-pair recall of the primary disjunctive blocking strategy is **79.25%**, with an entity-level coverage of **96.16%** and an average of **37.3 candidates per query** (out of a 40,000 target universe).

---

## 2. Evaluation Methodology Audit Findings

### A. S2 Selection
The original harness sequentially read the first 200,000 rows of `train_source2.tsv` (~3.9% of the 5.1M dataset).

### B. S3 Selection
The original harness sequentially read the first 200,000 rows of `train_source3.tsv` (~3.9% of the 5.1M dataset).

### C. Retrieval Universe Completeness
**No.** The original target index contained only ~3.9% of all S2/S3 records. Ground-truth matches for the sampled 5,000 S1 queries are distributed uniformly across all 5.1M rows. Consequently, ~97% of the true target records were completely absent from the indexed retrieval universe.

### D. Recall Denominator Flaw
**Yes.** The original evaluator evaluated candidate recall over all 17,362 ground-truth pairs for the 5,000 S1 queries, even though ~97% of those target records did not exist in the loaded index. A search index cannot retrieve records that were never loaded into it.

### E. Validity of the Initial 0.72% - 2.90% Recall
**Invalid.** The initial recall simply reflected the ~3.9% target sampling fraction (e.g. $0.79 \times 0.039 \approx 0.0308$, or ~3.0%). It measured target dataset slice intersection, not blocking capability.

### F. S1 Coverage Calculation
The initial S1 coverage suffered from the identical flaw, as it measured S1 entities for which at least one match was found out of ground-truth pairs whose targets were predominantly missing from the index.

### G. Candidate Counts
Candidate counts were computed on the 400k index (mean candidates 46-50, capped at `top_k=50`). While the candidate count per query was bounded, the precision was artificially deflated.

### H. Leakage and Sampling Bias
There was no train/test data leakage (evaluation strictly used `train_*.tsv`). However, there was severe target sampling omission bias.

---

## 3. Corrected Evaluation Methodology

1. **Deterministic S1 Sampling:** 500 S1 queries are sampled deterministically (seed 42).
2. **Guaranteed Positive Inclusion:** For all sampled S1 entities, all ground-truth S2 and S3 target IDs are identified and guaranteed to be indexed.
3. **Controlled Negative Distractors:** Additional negative target records from S2 and S3 are streamed and indexed to build a realistic 40,000-entity retrieval universe (20,000 S2 + 20,000 S3).
4. **Strict Denominator Assertions:** True-pair recall and entity coverage are computed strictly over ground-truth pairs whose target records are verified to be present in the evaluated retrieval universe.
5. **No Data Leakage:** Evaluated entirely within the training split without inspecting `test_*.tsv`.

---

## 4. Benchmark Results Comparison

| Strategy | True-Pair Recall | S2 Recall | S3 Recall | S1 Entity Coverage | Mean Cands/S1 | Median Cands | P95 Cands | Max Cands | Runtime (sec) | QPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **exact_name** | 20.75% | 19.53% | 21.89% | 55.65% | 0.8 | 1.0 | 2.0 | 6 | 0.0026s | 192,308 |
| **rare_tokens** | 78.52% | 78.25% | 78.76% | 94.67% | 38.0 | 50.0 | 50.0 | 50 | 0.0669s | 7,474 |
| **disjunctive_union** | **79.25%** | **79.18%** | **79.31%** | **96.16%** | **37.3** | **48.0** | **50.0** | **50** | **0.0647s** | **7,728** |
| **composite_union** | 75.70% | 76.14% | 75.30% | 93.82% | 46.7 | 48.0 | 50.0 | 50 | 0.0848s | 5,896 |

---

## 5. Key Diagnoses & Recommendations for Next Phase

1. **Current Best Strategy:** `disjunctive_union` achieves the highest recall (79.25%) and coverage (96.16%) with lower average candidate explosion (37.3 cands/query) and high throughput (7,728 QPS).
2. **Analysis of the ~20% Unrecovered Matches:**
   - Typographical differences and character-level spelling variations in company names (e.g. abbreviations, prefixes).
   - Entities whose names share no exact rare tokens with the target record (e.g. branch names vs parent names, or legal names vs trade names).
3. **Next Steps (Candidate Generation Phase):**
   - Incorporate character n-gram / fuzzy prefix indexing (e.g. 3-gram inverted index for terms with low token overlap).
   - Add country/jurisdiction-aware multi-attribute blocking (address tokens, postcodes, phone prefixes) before feature extraction.
   - Target $\ge 95\%$ candidate recall at candidate budget $\le 100$ per S1 query before proceeding to ML reranking.
