# MASTER PROMPT v2: Amazon Business Entity Resolution Challenge (Precision-Optimized, Ensemble Architecture)

## 0. Realistic Target Statement (Read First)

This prompt intentionally does **not** target "accuracy = 0.99." That is the wrong metric for this task.

- The official metric is **F0.5** (precision-weighted), evaluated **per-S1-entity**, not raw accuracy.
- Business entity resolution across noisy, multilingual, cross-source records is inherently imperfect. Realistic strong performance is generally in the **F0.5 ≈ 0.75–0.92** range depending on noise severity, not 0.99.
- Any AI implementing this must **never fabricate matches, force outputs toward a target score, or overfit thresholds to a single validation split** to chase an arbitrary number.
- The actual goal: **the highest F0.5 that is honestly supported by the data**, achieved through better blocking recall, better calibrated features, ensembled models, and correct threshold/entity-level decision logic — not by gaming the metric.

If the AI implementing this ever finds itself inventing data, silently relaxing validation rigor, or hard-coding results to hit a target score, it must stop and flag this instead.

---

## Role

Act as a senior Machine Learning Engineer, Entity Resolution / Record Linkage specialist, and production-grade data science architect.

You are reviewing and improving an implementation for the **Amazon Business Entity Resolution Challenge**. Your task is not to produce a generic entity-matching script. You must inspect the existing project, understand the challenge constraints, validate the current approach, identify weaknesses, and rebuild/improve the solution wherever technically justified — using an **ensembled, calibrated, explainable** architecture rather than a single default model.

The final implementation must be accurate, reproducible, explainable, computationally practical, and compliant with every competition rule.

---

# 1. Challenge Objective

Resolve business entities across three independent sources containing noisy and inconsistent business records.

- **Source 1 (S1):** deduplicated reference entities.
- **Source 2 (S2):** noisy business records.
- **Source 3 (S3):** noisy business records.

For every S1 entity, identify **all matching S2 and S3 records**.

A Source 1 entity may have:
- zero matches,
- one match,
- or multiple matches.

The system must support both one-to-one and one-to-many relationships, and must explicitly support **zero matches (singletons)** without being forced to output something.

---

# 2. Mandatory Input Schema

Each source contains:

```text
entity_id
business_name
business_address
country
```

Entity ID prefixes indicate the source: `S1-...`, `S2-...`, `S3-...`

Training countries include US and India; **test data also contains France**. Country handling must be **open-set** — never hard-code a country list. Detect countries dynamically from the data at runtime.

---

# 3. Data Loading Requirements

All source files are TSV. Load with:

```python
pd.read_csv(path, sep="\t")
```

Never assume comma delimiters (addresses/ID lists may contain commas).

Expected training files:
```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
```

Expected test files:
```text
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

Before implementing the pipeline, run a full profiling pass and inspect:
- row counts, column names, null rates
- duplicate IDs, duplicate business records
- country distribution
- name/address length distributions
- multilingual content share
- corrupted/malformed values
- exact and near duplicates

Do not assume sample rows represent the full dataset distribution. Produce a written data profile before writing pipeline code.

---

# 4. Ground Truth

Expected schema:
```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` is comma-separated S2/S3 IDs. Empty = no match.

**Verify the actual file schema before proceeding.** If it doesn't match, stop and report — do not silently reinterpret it.

Build training pair labels strictly from verified ground truth:
```text
S1 + S2/S3 candidate pair → MATCH / NON-MATCH
```

Never train on guessed or inferred labels.

---

# 5. Observed Data Characteristics — Name Noise

Includes: capitalization differences, punctuation differences, extra spaces, legal suffixes, abbreviations, duplicated words, spelling errors, character corruption, inserted/deleted characters, DBA/trade names, multilingual names, mixed-language names, transliteration.

Examples: `Fanni's Dmaigesostcis`, `Animal Hanisch Hospirlg`, and multilingual records in Hindi/Tamil/mixed script.

The system must **not** assume English-only text and must **not** strip non-ASCII characters. All text processing must be Unicode-aware (NFC/NFKC normalization, not ASCII transliteration by default).

---

# 6. Address Noise

Includes: different component ordering, abbreviations, missing components, duplicated components, punctuation/casing differences, transliteration, locality variation, landmarks, unit/apartment variation, house/plot numbers, postal codes, malformed/null values.

**Address matching must be robust but not mandatory.** The system must gracefully degrade to name-heavy matching when address data is missing, using a `missing_address_flag` feature rather than treating missing address as a mismatch signal.

---

# 7. Core Pipeline (Upgraded)

```text
Raw Data
   ↓
Data Validation / Profiling
   ↓
Unicode-Aware Normalization
   ↓
Multi-Strategy Blocking (token index + char n-gram LSH + ANN retrieval)
   ↓
Candidate Pair Dataset  (this is what candidate_pairs.tsv reflects)
   ↓
Pairwise Feature Engineering (lexical + phonetic + numeric + cross-field)
   ↓
Stacked Ensemble Match Model (LightGBM + XGBoost + CatBoost → meta-learner)
   ↓
Isotonic Probability Calibration
   ↓
Per-S1 Threshold / Entity-Level Decision Optimization for macro F0.5
   ↓
Final Match Selection (subset of candidates)
   ↓
matching_results.tsv
candidate_pairs.tsv
   ↓
Submission Validator
```

Do not skip candidate generation. Do not skip calibration — raw booster scores are not well-calibrated probabilities and directly distort threshold selection.

---

# 8. Candidate Generation / Blocking (Upgraded)

Blocking is the most important recall bottleneck in this challenge. Use a **union of complementary retrieval strategies**, not a single blocking key.

### 8.1 Country blocking
Prefer same-country candidates, but do not hard-discard cross-country pairs — validate empirically whether noisy data produces legitimate cross-country matches (e.g., mislabeled country fields) before excluding them.

### 8.2 Exact/token blocking
- normalized name prefix / first token
- rare-token inverted index
- token signatures

### 8.3 Character n-gram retrieval (upgraded)
- Build a character 3–5-gram TF-IDF index per source.
- Use **MinHash LSH** (via `datasketch`) for near-duplicate name retrieval at scale — far cheaper than brute-force cosine similarity for large datasets, while preserving high recall on typos/corruption.

### 8.4 Approximate Nearest Neighbor retrieval (new)
- Encode normalized names (and optionally name+address) into TF-IDF or character-ngram vector space.
- Use **FAISS** (`IndexFlatIP` for small data, `IndexIVFFlat`/HNSW for larger data) for fast top-K approximate retrieval per S1 entity against S2/S3.
- This catches severe misspellings and reordering that token-based blocking misses, at sub-quadratic cost.

### 8.5 Address-based blocking
- city/locality tokens, state/province tokens, postal codes, house/plot numbers
- character n-gram / TF-IDF retrieval on normalized address string

### 8.6 Hybrid blocking
Union outputs of:
```text
country + name token
country + address token
character n-gram LSH
ANN vector retrieval (name)
ANN vector retrieval (name+address)
```

**Candidate recall is the top priority at this stage** — measure and report recall@K on the training ground truth before building any features. A blocking recall below ~98% on validation caps the achievable F0.5 regardless of downstream model quality; report this number explicitly.

---

# 9. Candidate Set Requirement

`candidate_pairs.tsv` MUST represent the exact final candidate list fed into the ML matcher — not an earlier, larger intermediate set. If the pipeline does:

```text
5,000 initial candidates → 300 refined candidates → ML model
```

then `candidate_pairs.tsv` contains the 300, and **final predicted matches must be a strict subset of it.**

---

# 10. Feature Engineering — Name Features

- normalized exact match
- token overlap / token-set (Jaccard) similarity
- character n-gram cosine similarity (TF-IDF)
- Levenshtein / normalized edit similarity
- Jaro-Winkler similarity
- longest common subsequence ratio
- length difference / length ratio
- common token count, rare-token overlap
- legal-suffix-normalized similarity (strip Inc/LLC/Pvt Ltd/Pte etc. before comparing, but also keep a suffix-match flag as a separate feature)
- **phonetic similarity** (Double Metaphone / Soundex) as a supplementary signal, not primary — validate before keeping
- multilingual-safe character n-gram similarity (script-agnostic; do not depend on Latin-only tokenization)

Use ablation/feature-importance testing (SHAP or permutation importance) to prune features that don't measurably help. Do not include every feature blindly.

---

# 11. Feature Engineering — Address Features

- normalized exact match, token overlap, Jaccard, TF-IDF cosine, character n-gram cosine, edit similarity
- city match, state/province match, postal code match
- house/plot number match, apartment/unit number match
- numeric token overlap (treat numbers carefully — a shared postal code is strong evidence; a shared generic small number like "1" is weak)
- landmark/locality token overlap
- address length difference
- `missing_address_flag` (both/one/neither address present)

---

# 12. Cross-Field & Interaction Features

```text
country_exact_match
name_address_joint_similarity
name_length_ratio
address_length_ratio
number_overlap
city_overlap / state_overlap
token_count_difference
missing_name_flag / missing_address_flag
```

Interaction buckets (validate each):
```text
high_name_sim + high_address_sim   → strong positive signal
high_name_sim + low_address_sim    → ambiguous, check for common-name collisions
low_name_sim  + high_address_sim   → possible DBA/trade-name case, worth extra scrutiny
```

---

# 13. Multilingual Processing

Dataset contains English, Hindi, Tamil, and French text (and mixed-script records).

Requirements:
- preserve Unicode; normalize with NFC/NFKC (not ASCII stripping)
- tokenize in a script-agnostic way (don't assume whitespace-only English tokenization works for all scripts)
- character n-gram similarity is the most robust cross-script baseline — prioritize it
- **optional**: a multilingual sentence-embedding reranker (e.g., `LaBSE` or `intfloat/multilingual-e5-base`) applied only to the top-K ambiguous candidates from blocking, gated strictly behind measured validation F0.5 improvement vs. its added latency/memory cost
- do not add a multilingual LLM/embedding model by default — justify with numbers

---

# 14. Model Selection — Stacked Ensemble (Upgraded Recommendation)

Do not settle on a single booster by default. Benchmark and then combine.

### 14.1 Base learners to benchmark
1. Logistic Regression (baseline, interpretable, fast)
2. LightGBM
3. XGBoost
4. CatBoost (particularly strong here since it handles categorical-like flags and sparse similarity features natively, and tends to be robust on smaller noisy tabular sets)
5. HistGradientBoostingClassifier (dependency-light fallback)

### 14.2 Recommended final architecture: Stacked Ensemble
```text
LightGBM  ─┐
XGBoost   ─┼─► out-of-fold predictions ─► Logistic Regression meta-learner ─► final probability
CatBoost  ─┘
```

Why a stack over a single booster:
- the three boosters have different split heuristics and regularization behavior, so their errors are only partially correlated — stacking typically yields a measurable F0.5 lift (commonly 1–3 points) over the single best base model on tabular similarity-feature problems
- a linear meta-learner on 3 well-behaved probability inputs is cheap, fast to train, and does not meaningfully hurt reproducibility or interpretability
- if stacking does **not** show a validated improvement over the single best booster on your actual data, use the single best booster instead — do not add the stack for its own sake

### 14.3 Calibration (mandatory)
Apply **isotonic regression** calibration (fit on a held-out calibration fold, not the training fold) to the final ensemble output before any thresholding. Boosted-tree outputs are frequently overconfident/underconfident at the tails, and this directly affects macro F0.5 optimization in Section 19.

### 14.4 When to consider more
A cross-encoder or small local LLM may be evaluated **only** for a narrow, genuinely ambiguous residual subset (e.g., near-threshold cases where the ensemble's margin is small) if it:
- demonstrably improves validation F0.5 on that subset,
- respects the competition's model size/license constraints (≤8B parameters, no external data),
- stays within runtime/memory budgets,
- remains reproducible with a fixed seed/version.

Do not make an LLM the default or primary matcher.

---

# 15. Retrieval + Reranking Architecture

```text
Blocking / Retrieval  (maximize recall)
        ↓
Top-K candidate generation
        ↓
Feature-based ensemble reranker  (maximize precision)
        ↓
Calibration
        ↓
Entity-level decision
```

Never do a full O(N²) S1×S2/S3 comparison if avoidable — always retrieve top-K first.

---

# 16. Model Selection Must Be Evidence-Based

Report a comparison table like:

| Model | F0.5 | Precision | Recall | Candidate Recall | Train Time | Inference Time | Memory |
|---|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | | |
| LightGBM | | | | | | | |
| XGBoost | | | | | | | |
| CatBoost | | | | | | | |
| Stacked Ensemble | | | | | | | |

Select based on measured numbers, not assumption. If a simpler model matches the ensemble within noise, prefer the simpler model for reproducibility.

---

# 17. Validation Strategy

- Never tune thresholds on the test set.
- Use an **entity-level split** (split by S1 entity, not by pair) to prevent leakage — near-duplicate records of the same underlying entity must not appear in both train and validation.
- Prefer k-fold (e.g., 5-fold) entity-level cross-validation for model comparison and out-of-fold predictions needed for stacking.
- Use a dedicated calibration fold, separate from the fold(s) used for threshold selection.

---

# 18. F0.5 Optimization — Correct Metric Implementation

The official metric is **F0.5**, macro-averaged **per S1 entity** — this is not the same as global pairwise F0.5. Implement the exact official evaluation logic (or the closest verifiable approximation) before optimizing anything.

- Never use a fixed `threshold = 0.5` default.
- Sweep thresholds (e.g., 0.10 → 0.95 in fine steps) on the calibration/validation split and select the threshold maximizing **per-entity macro F0.5**, not pairwise F0.5 — these can select different optimal thresholds.
- Re-verify the chosen threshold on a separate validation fold before finalizing (avoid single-split overfitting).

---

# 19. Singleton Handling

S1 entities with no true match must be allowed to output `matched_entity_ids = empty`. Never force at least one match. Because F0.5 is precision-heavy, a false positive on a true singleton is disproportionately costly — bias the entity-level decision toward abstaining when evidence is weak.

---

# 20. One-to-Many Matching

Do not hard-limit to top-1 candidate. Evaluate each candidate independently against the calibrated threshold, but also consider (validate empirically before adopting):
- absolute candidate score
- margin from the next-best non-selected candidate
- number of strong corroborating fields (name AND address both strong vs. only one)

Add weak "plausible" matches only if validation shows it doesn't hurt precision enough to reduce F0.5.

---

# 21. Two-Stage Decision (Mandatory, not optional)

### Stage 1 — Pairwise probability
`P(match | S1, candidate)` from the calibrated stacked ensemble.

### Stage 2 — Entity-level decision
Apply, in order:
1. Threshold from Section 18.
2. Margin/consistency check across multiple candidates of the same S1.
3. Duplicate-ID conflict resolution (a single S2/S3 record should not be assigned to multiple S1 entities unless the data genuinely supports it — validate this constraint against training ground truth).

Only keep additional post-processing steps that show a measured official-metric improvement.

---

# 22. Leakage Prevention

Never use: test ground truth, external business databases, external geocoding, internet lookups, commercial entity-resolution APIs, government registries, external directories, or any pretrained embedding model that was fine-tuned on this specific dataset externally. All information must come from the provided dataset only.

---

# 23. Competition Constraints

- Exact output format compliance
- Only S2/S3 IDs may be matched
- Every S1 test entity appears exactly once
- No duplicate matched IDs
- Singletons represented correctly (empty `matched_entity_ids`)
- Final matches must be a subset of candidates
- Model must satisfy license/parameter constraints — **max 8B parameters**
- No prohibited external data lookup

---

# 24. Required Output

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

### matching_results.tsv
```text
source1_entity_id    matched_entity_ids
```
No match: `S1-12345` with empty `matched_entity_ids`.
Multiple matches: `S1-12345    S2-111,S2-222,S3-333`
Deterministic ordering required (e.g., sort matched IDs lexicographically).

### candidate_pairs.tsv
Must equal the exact final candidate set fed to the ML matcher. Final predictions must be a subset of it. Deterministic output required.

---

# 25. Submission Validation

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Implementation is not complete until this passes.

---

# 26. Reproducibility

```text
code/business_entity_resolution/src/
README.md
requirements.txt
Documentation_template.md
output/
```

Use deterministic random seeds (fix seeds for numpy, LightGBM, XGBoost, CatBoost, FAISS index build order), config files/constants for all hyperparameters and thresholds, and documented preprocessing/feature/model/threshold/blocking parameters. No hard-coded machine-specific paths.

---

# 27. Computational Efficiency

Before implementing expensive steps, measure row counts, candidate pair counts, memory, and runtime. Use vectorization, inverted indexes, MinHash LSH, FAISS ANN indexes, sparse matrices, caching, batching, and multiprocessing only where it measurably helps. Never generate an unnecessary full Cartesian product.

---

# 28. Explainability

Retain per-candidate debug info:
```text
name_similarity
address_similarity
country_match
numeric_overlap
city_match
state_match
base_model_probabilities (LGBM, XGB, CatBoost)
ensemble_probability
calibrated_probability
decision_threshold
margin_to_next_candidate
```

Additionally generate **SHAP values** for the final model on a sample of predictions (especially false positives/negatives) to explain which features drove each decision — this is far more actionable for debugging than raw feature values alone.

---

# 29. Error Analysis

### False positives — inspect especially:
similar names but different addresses; same city, different business; common/generic words; duplicated company names; legal-suffix-only similarity.

### False negatives — inspect especially:
severe spelling corruption; multilingual names; missing addresses; DBA names; transliteration; reordered addresses; abbreviations.

Feed findings back into blocking and feature design — this is an iterative loop, not a one-time pass.

---

# 30. Ablation Testing

```text
Baseline: name + address exact features
+ edit similarity
+ character n-grams
+ TF-IDF
+ address numeric features
+ phonetic features
+ multilingual embedding reranker
+ stacked ensemble vs. single best booster
+ isotonic calibration vs. uncalibrated
```

Measure actual validation F0.5 impact of each addition. Drop anything without measurable benefit.

---

# 31. Important Instruction to the Implementing AI

You are explicitly authorized and expected to improve on this specification further if evidence supports it — better blocking, better similarity functions, better features, better models, better retrieval, better multilingual strategy, better validation design, better thresholding/calibration, better entity-level decisions, better optimization.

Do not blindly follow the stacked-ensemble recommendation in Section 14 either — if a single booster or a different architecture is demonstrably better on **this actual dataset**, use it and explain why with numbers.

**Never optimize toward a specific target score (e.g., "0.99") by fabricating matches, cherry-picking a favorable split, or overfitting the threshold to one validation run.** The goal is the honestly-best F0.5 achievable, with recall, reproducibility, and compliance preserved.

---

# 32. Critical Review Requirement (Before Changing Code)

Identify explicitly:
1. What is currently correct?
2. What is currently incorrect?
3. What is inefficient?
4. What can cause false positives?
5. What can cause false negatives?
6. What violates the challenge specification?
7. What causes data leakage?
8. What prevents reproducibility?
9. What prevents good F0.5?
10. What should be replaced rather than patched?

Do not preserve a bad architecture merely because it already exists.

---

# 33. Do Not Guess Missing Information

If a file, schema, dataset property, or ground-truth format has not been verified: inspect the actual file, report uncertainty, do not invent values, do not fabricate matches. In particular, verify `train_ground_truth.tsv`'s actual schema before building labels.

---

# 34. Deliverables

### A. Technical Assessment — current architecture and its weaknesses
### B. Recommended Architecture — final improved architecture with justification per component
### C. Model Selection — comparison table (Section 16) + final choice with reasoning
### D. Blocking Strategy — every method used + measured candidate recall
### E. Feature Engineering — final feature set + ablation results
### F. Validation — split strategy, leakage prevention, F0.5 calculation method, threshold tuning, model comparison
### G. Implementation — the actual improved pipeline code
### H. Outputs — `matching_results.tsv`, `candidate_pairs.tsv`
### I. Documentation — updated README + methodology doc
### J. Reproducibility — exact commands to install, train, validate, predict, and validate submission

---

# 35. Final Quality Gate

- [ ] Dataset loaded with TSV delimiter
- [ ] Actual schemas verified (not assumed)
- [ ] Ground truth schema verified
- [ ] No external data used
- [ ] Unicode preserved throughout
- [ ] Country handling is open-set
- [ ] Missing addresses handled gracefully
- [ ] Multi-strategy blocking implemented (token + LSH + ANN)
- [ ] Candidate recall measured and reported (target ≥98%)
- [ ] `candidate_pairs.tsv` matches exact final ML candidate set
- [ ] Model benchmark table completed with real numbers
- [ ] Stacked ensemble validated against single-best-booster baseline
- [ ] Isotonic calibration applied and validated
- [ ] Entity-level (not pair-level) train/validation split used
- [ ] Official per-entity macro F0.5 correctly implemented
- [ ] Threshold optimized on validation, re-checked on a separate fold
- [ ] Singleton handling implemented and tested
- [ ] One-to-many matching supported and validated
- [ ] Final matches are a strict subset of candidates
- [ ] Every test S1 entity appears exactly once
- [ ] No duplicate matched IDs
- [ ] Only valid S2/S3 IDs output
- [ ] SHAP/explainability artifacts generated for error analysis
- [ ] False positive / false negative analysis documented
- [ ] Ablation results documented with numbers
- [ ] Validator script passes
- [ ] Runtime and memory are reasonable for actual dataset size
- [ ] Results are fully deterministic (fixed seeds everywhere)
- [ ] README/documentation complete

---

# 36. Final Instruction

Treat this as a serious competitive Entity Resolution / Record Linkage system, not a toy fuzzy-matching script.

Do not optimize for code simplicity at the expense of official F0.5.
Do not optimize for recall at the expense of precision (F0.5 punishes false positives harder).
Do not optimize for model sophistication at the expense of reproducibility.
**Do not optimize toward a fabricated target score at the expense of honesty.**

Final objective, in order:

**Blocking (high recall) → Candidate Retrieval → Pairwise Feature Engineering → Calibrated Stacked-Ensemble Reranking → Per-Entity F0.5-Optimized Decision → Validated, Reproducible Submission**

If analysis of the actual dataset shows a different architecture is superior, modify this plan and clearly explain why with measured evidence. Always prioritize measured validation performance and competition compliance over assumptions — and over any externally imposed target score.
